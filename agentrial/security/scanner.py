"""MCP Security Scanner.

Analyzes MCP server configurations and tool definitions for known
vulnerability classes:

1. Prompt injection via tool descriptions
2. Tool shadowing (near-duplicate names of trusted tools)
3. Data exfiltration via dangerous tool combinations
4. Permission escalation patterns
5. Rug pull detection (capability mismatch)

Usage:
    scanner = MCPSecurityScanner()
    result = scanner.scan(mcp_config)
    print(f"Score: {result.score}/10 — {len(result.findings)} issues found")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(Enum):
    """Severity level of a security finding."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityFinding:
    """A single security finding from the scan."""

    severity: Severity
    category: str
    title: str
    description: str
    tool_name: str | None = None
    server_name: str | None = None
    recommendation: str = ""
    evidence: str = ""


@dataclass
class ScanResult:
    """Result of an MCP security scan."""

    findings: list[SecurityFinding] = field(default_factory=list)
    tools_scanned: int = 0
    servers_scanned: int = 0

    @property
    def score(self) -> float:
        """Security score from 0-10 (10 = no issues found)."""
        if not self.findings:
            return 10.0

        deductions = {
            Severity.CRITICAL: 3.0,
            Severity.HIGH: 2.0,
            Severity.MEDIUM: 1.0,
            Severity.LOW: 0.5,
            Severity.INFO: 0.0,
        }

        total_deduction = sum(
            deductions.get(f.severity, 0) for f in self.findings
        )
        return max(0.0, 10.0 - total_deduction)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def passed(self) -> bool:
        """Whether the scan passed (no critical or high findings)."""
        return self.critical_count == 0 and self.high_count == 0


# --- Injection patterns ---

# Patterns that indicate prompt injection in tool descriptions
INJECTION_PATTERNS = [
    # Direct instructions
    (r"(?i)\balways\b.{0,50}\b(include|add|append|send|return)\b", "instruction_injection"),
    (r"(?i)\bnever\b.{0,50}\b(tell|reveal|mention|show)\b", "suppression_injection"),
    (r"(?i)\bignore\b.{0,30}\b(previous|prior|above|other)\b", "override_injection"),
    (r"(?i)\byou\s+(must|should|need\s+to|have\s+to)\b", "directive_injection"),
    (r"(?i)\bdo\s+not\b.{0,30}\b(ask|check|verify|validate)\b", "bypass_injection"),
    # Data exfiltration triggers
    (r"(?i)(ssh|private|secret|password|token|key|credential)", "sensitive_data_reference"),
    (r"(?i)(\.env|\.ssh|\.aws|\.git/config|id_rsa)", "sensitive_file_reference"),
    # System prompt manipulation
    (r"(?i)\bsystem\s*prompt\b", "system_prompt_reference"),
    (r"(?i)\b(override|bypass|circumvent)\b.{0,30}\b(safety|restriction|limit)\b", "safety_bypass"),
]

# Well-known tool names that could be targets for shadowing
KNOWN_TOOLS = {
    "read_file", "write_file", "list_files", "search_files",
    "run_command", "execute", "exec", "shell",
    "send_email", "send_message", "http_request",
    "read_database", "query_database", "sql_query",
    "get_secret", "get_env", "environment",
}

# Dangerous tool combinations
DANGEROUS_COMBINATIONS = [
    ({"read_file", "read_database", "get_secret", "get_env"},
     {"send_email", "http_request", "send_message", "write_file"},
     "Data exfiltration: read + send tools enable unauthorized data transfer"),
    ({"run_command", "execute", "exec", "shell"},
     {"read_file", "write_file"},
     "Code execution + file access enables arbitrary system manipulation"),
    ({"get_secret", "get_env"},
     {"http_request", "send_email", "send_message"},
     "Secret access + network tools enable credential exfiltration"),
]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(
                min(
                    curr_row[j] + 1,       # insert
                    prev_row[j + 1] + 1,   # delete
                    prev_row[j] + cost,     # replace
                )
            )
        prev_row = curr_row

    return prev_row[-1]


class MCPSecurityScanner:
    """Scanner for MCP server security vulnerabilities.

    Analyzes tool definitions, descriptions, and configurations
    for known vulnerability patterns.
    """

    def __init__(
        self,
        custom_trusted_tools: set[str] | None = None,
        custom_injection_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        """Initialize the scanner.

        Args:
            custom_trusted_tools: Additional tool names to check for shadowing.
            custom_injection_patterns: Additional (regex, category) patterns.
        """
        self.trusted_tools = KNOWN_TOOLS.copy()
        if custom_trusted_tools:
            self.trusted_tools |= custom_trusted_tools

        self.injection_patterns = list(INJECTION_PATTERNS)
        if custom_injection_patterns:
            self.injection_patterns.extend(custom_injection_patterns)

    def scan(self, config: dict[str, Any]) -> ScanResult:
        """Scan an MCP configuration for security issues.

        Args:
            config: MCP configuration dict. Expected format:
                {
                    "mcpServers": {
                        "server-name": {
                            "command": "...",
                            "args": [...],
                            "env": {...},
                            "tools": [
                                {
                                    "name": "tool_name",
                                    "description": "...",
                                    "inputSchema": {...}
                                }
                            ]
                        }
                    }
                }

        Returns:
            ScanResult with findings and score.
        """
        result = ScanResult()

        servers = config.get("mcpServers", config.get("servers", {}))
        result.servers_scanned = len(servers)

        all_tools: list[dict[str, Any]] = []
        tool_server_map: dict[str, str] = {}

        for server_name, server_config in servers.items():
            server_tools = server_config.get("tools", [])

            # Check server-level security
            self._check_server_config(server_name, server_config, result)

            for tool in server_tools:
                tool_name = tool.get("name", "")
                all_tools.append(tool)
                tool_server_map[tool_name] = server_name
                result.tools_scanned += 1

                # Check each tool
                self._check_injection(tool, server_name, result)
                self._check_shadowing(tool, server_name, result)
                self._check_permission_escalation(tool, server_name, result)

        # Cross-tool checks
        self._check_dangerous_combinations(all_tools, tool_server_map, result)
        self._check_rug_pull(servers, result)

        return result

    def _check_injection(
        self,
        tool: dict[str, Any],
        server_name: str,
        result: ScanResult,
    ) -> None:
        """Check tool description for prompt injection patterns."""
        description = tool.get("description", "")
        tool_name = tool.get("name", "unknown")

        for pattern, category in self.injection_patterns:
            match = re.search(pattern, description)
            if match:
                # Determine severity based on category
                if category in ("sensitive_file_reference", "safety_bypass"):
                    severity = Severity.CRITICAL
                elif category in (
                    "instruction_injection",
                    "override_injection",
                    "bypass_injection",
                ):
                    severity = Severity.HIGH
                elif category in ("sensitive_data_reference", "system_prompt_reference"):
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.MEDIUM

                result.findings.append(
                    SecurityFinding(
                        severity=severity,
                        category="prompt_injection",
                        title=f"Tool description contains {category.replace('_', ' ')}",
                        description=(
                            f"Tool '{tool_name}' on server '{server_name}' has a "
                            f"description that matches injection pattern: {category}"
                        ),
                        tool_name=tool_name,
                        server_name=server_name,
                        recommendation=(
                            "Review the tool description and remove any instructions "
                            "that could alter agent behavior."
                        ),
                        evidence=match.group(0),
                    )
                )

    def _check_shadowing(
        self,
        tool: dict[str, Any],
        server_name: str,
        result: ScanResult,
    ) -> None:
        """Check if a tool name shadows a known trusted tool."""
        tool_name = tool.get("name", "")
        if not tool_name:
            return

        for trusted in self.trusted_tools:
            if tool_name == trusted:
                continue  # Exact match is fine (legitimate implementation)

            distance = _levenshtein_distance(tool_name.lower(), trusted.lower())

            # Flag if edit distance is 1-2 (likely typosquat)
            if 0 < distance <= 2 and len(tool_name) >= 4:
                result.findings.append(
                    SecurityFinding(
                        severity=Severity.HIGH,
                        category="tool_shadowing",
                        title=f"Tool name '{tool_name}' shadows '{trusted}'",
                        description=(
                            f"Tool '{tool_name}' on server '{server_name}' has a name "
                            f"very similar to trusted tool '{trusted}' "
                            f"(edit distance: {distance}). This could be an attempt "
                            f"to intercept calls meant for the trusted tool."
                        ),
                        tool_name=tool_name,
                        server_name=server_name,
                        recommendation=(
                            f"Verify this tool is legitimate and not attempting to "
                            f"shadow '{trusted}'. Consider renaming to avoid confusion."
                        ),
                        evidence=f"'{tool_name}' vs '{trusted}' (distance={distance})",
                    )
                )

    def _check_permission_escalation(
        self,
        tool: dict[str, Any],
        server_name: str,
        result: ScanResult,
    ) -> None:
        """Check for permission escalation patterns in tool schemas."""
        tool_name = tool.get("name", "")
        schema = tool.get("inputSchema", {})
        description = tool.get("description", "").lower()

        # Check if a seemingly simple tool has overly broad capabilities
        properties = schema.get("properties", {})

        # Flag tools with shell/command execution capability
        for prop_name, prop_def in properties.items():
            prop_desc = str(prop_def.get("description", "")).lower()
            if any(
                kw in prop_name.lower() or kw in prop_desc
                for kw in ("command", "shell", "exec", "eval", "script")
            ):
                if not any(
                    kw in tool_name.lower()
                    for kw in ("command", "shell", "exec", "run")
                ):
                    result.findings.append(
                        SecurityFinding(
                            severity=Severity.HIGH,
                            category="permission_escalation",
                            title=f"Hidden command execution in '{tool_name}'",
                            description=(
                                f"Tool '{tool_name}' on server '{server_name}' "
                                f"has a parameter '{prop_name}' that appears to "
                                f"enable command execution, but the tool name does "
                                f"not indicate this capability."
                            ),
                            tool_name=tool_name,
                            server_name=server_name,
                            recommendation=(
                                "Verify this tool's actual capabilities match its "
                                "declared purpose. Hidden execution capabilities "
                                "are a security risk."
                            ),
                            evidence=f"Parameter '{prop_name}' in tool '{tool_name}'",
                        )
                    )

        # Check for filesystem access in non-file tools
        if "file" not in tool_name.lower() and "fs" not in tool_name.lower():
            if any(
                kw in description
                for kw in ("read file", "write file", "filesystem", "file system")
            ):
                result.findings.append(
                    SecurityFinding(
                        severity=Severity.MEDIUM,
                        category="permission_escalation",
                        title=f"Undeclared filesystem access in '{tool_name}'",
                        description=(
                            f"Tool '{tool_name}' on server '{server_name}' "
                            f"description mentions filesystem access but the "
                            f"tool name does not indicate file operations."
                        ),
                        tool_name=tool_name,
                        server_name=server_name,
                        recommendation=(
                            "Make tool capabilities explicit in the tool name."
                        ),
                    )
                )

    def _check_dangerous_combinations(
        self,
        tools: list[dict[str, Any]],
        tool_server_map: dict[str, str],
        result: ScanResult,
    ) -> None:
        """Check for dangerous combinations of tools across servers."""
        tool_names = {t.get("name", "").lower() for t in tools}

        for read_group, send_group, reason in DANGEROUS_COMBINATIONS:
            read_matches = tool_names & {t.lower() for t in read_group}
            send_matches = tool_names & {t.lower() for t in send_group}

            if read_matches and send_matches:
                # Check if they come from different servers (higher risk)
                read_servers = {
                    tool_server_map.get(t, "unknown") for t in read_matches
                }
                send_servers = {
                    tool_server_map.get(t, "unknown") for t in send_matches
                }

                cross_server = read_servers != send_servers
                severity = Severity.HIGH if cross_server else Severity.MEDIUM

                result.findings.append(
                    SecurityFinding(
                        severity=severity,
                        category="dangerous_combination",
                        title="Dangerous tool combination detected",
                        description=reason,
                        recommendation=(
                            "Review whether these tools need to coexist. "
                            "Consider adding access controls or scoping."
                        ),
                        evidence=(
                            f"Read tools: {sorted(read_matches)}, "
                            f"Send tools: {sorted(send_matches)}"
                        ),
                    )
                )

    def _check_rug_pull(
        self,
        servers: dict[str, Any],
        result: ScanResult,
    ) -> None:
        """Check for capability mismatches (rug pull detection).

        A rug pull is when a server's declared capabilities don't match
        what its tools actually do. This is detected by analyzing tool
        descriptions against the server's stated purpose.
        """
        for server_name, server_config in servers.items():
            tools = server_config.get("tools", [])
            server_desc = str(server_config.get("description", "")).lower()

            if not server_desc or not tools:
                continue

            # Check if any tools do things unrelated to the server description
            for tool in tools:
                tool_desc = tool.get("description", "").lower()
                tool_name = tool.get("name", "")

                # Flag network/send tools on a server that claims to be read-only
                if any(kw in server_desc for kw in ("read-only", "readonly", "read only")):
                    if any(
                        kw in tool_desc or kw in tool_name.lower()
                        for kw in ("write", "send", "post", "delete", "modify", "update")
                    ):
                        result.findings.append(
                            SecurityFinding(
                                severity=Severity.HIGH,
                                category="rug_pull",
                                title=f"Write capability on read-only server '{server_name}'",
                                description=(
                                    f"Server '{server_name}' claims to be read-only but "
                                    f"tool '{tool_name}' has write/send capabilities."
                                ),
                                tool_name=tool_name,
                                server_name=server_name,
                                recommendation=(
                                    "Verify server capabilities match its description. "
                                    "A server claiming read-only should not have write tools."
                                ),
                            )
                        )

    def _check_server_config(
        self,
        server_name: str,
        config: dict[str, Any],
        result: ScanResult,
    ) -> None:
        """Check server-level configuration for security issues."""
        env = config.get("env", {})

        # Check for secrets in environment variables
        for key, value in env.items():
            if isinstance(value, str) and len(value) > 10:
                # Check if it looks like a real secret (not a placeholder)
                if any(
                    kw in key.upper()
                    for kw in ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "PRIVATE_KEY")
                ):
                    result.findings.append(
                        SecurityFinding(
                            severity=Severity.MEDIUM,
                            category="credential_exposure",
                            title=f"Credential in server config '{server_name}'",
                            description=(
                                f"Server '{server_name}' has environment variable "
                                f"'{key}' that appears to contain a credential. "
                                f"Avoid storing secrets in plain text config files."
                            ),
                            server_name=server_name,
                            recommendation=(
                                "Use a secret manager or environment variable "
                                "references instead of inline credentials."
                            ),
                            evidence=f"env.{key} = {'*' * min(len(value), 8)}...",
                        )
                    )


def scan_mcp_config(
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    custom_trusted_tools: set[str] | None = None,
) -> ScanResult:
    """Scan an MCP configuration file or dict for security issues.

    Args:
        config_path: Path to MCP config JSON file.
        config: MCP config as a dict (alternative to file path).
        custom_trusted_tools: Additional trusted tool names.

    Returns:
        ScanResult with findings and security score.

    Raises:
        ValueError: If neither config_path nor config is provided.
        FileNotFoundError: If config_path doesn't exist.
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either config_path or config must be provided")

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"MCP config not found: {path}")

        with open(path) as f:
            config = json.load(f)

    scanner = MCPSecurityScanner(custom_trusted_tools=custom_trusted_tools)
    return scanner.scan(config)


def print_scan_result(
    result: ScanResult,
    console: Any = None,
) -> None:
    """Print scan results to terminal.

    Args:
        result: Scan result to display.
        console: Rich console (creates one if not provided).
    """
    from rich.console import Console
    from rich.table import Table

    if console is None:
        console = Console()

    # Header
    status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
    console.print(f"\n[bold]MCP Security Scan:[/bold] {status}")
    console.print(
        f"Servers: {result.servers_scanned} | "
        f"Tools: {result.tools_scanned} | "
        f"Score: {result.score:.1f}/10"
    )

    if not result.findings:
        console.print("[green]No security issues found.[/green]")
        return

    # Findings table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Severity", min_width=8)
    table.add_column("Category", min_width=15)
    table.add_column("Finding", min_width=40)
    table.add_column("Tool/Server", min_width=15)

    severity_colors = {
        Severity.CRITICAL: "red bold",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "dim",
        Severity.INFO: "blue",
    }

    for finding in sorted(result.findings, key=lambda f: list(Severity).index(f.severity)):
        color = severity_colors.get(finding.severity, "white")
        sev_text = f"[{color}]{finding.severity.value.upper()}[/{color}]"

        location = finding.tool_name or finding.server_name or "-"

        table.add_row(
            sev_text,
            finding.category,
            finding.title,
            location,
        )

    console.print(table)

    # Recommendations
    critical_findings = [
        f for f in result.findings
        if f.severity in (Severity.CRITICAL, Severity.HIGH)
    ]
    if critical_findings:
        count = len(critical_findings)
        console.print(
            f"\n[bold red]{count} critical/high findings "
            f"require attention:[/bold red]"
        )
        for f in critical_findings:
            console.print(f"  • {f.title}")
            if f.recommendation:
                console.print(f"    [dim]{f.recommendation}[/dim]")
