"""GitHub Actions integration for Agentrial."""

from typing import Any


def generate_action_yaml(
    threshold: float = 0.85,
    trials: int = 10,
    python_version: str = "3.11",
) -> str:
    """Generate a GitHub Actions workflow YAML for Agentrial.

    Args:
        threshold: Minimum pass rate to pass CI.
        trials: Number of trials per test.
        python_version: Python version to use.

    Returns:
        YAML content as a string.
    """
    return f"""name: Agentrial

on:
  pull_request:
    branches: [ main, master ]
  push:
    branches: [ main, master ]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '{python_version}'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install agentrial
        pip install -r requirements.txt

    - name: Run Agentrial
      run: |
        agentrial run --trials {trials} --threshold {threshold} --output results.json
      env:
        OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}

    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: agenteval-results
        path: results.json

    - name: Compare with baseline
      if: github.event_name == 'pull_request'
      continue-on-error: true
      run: |
        if [ -f baseline.json ]; then
          agentrial compare results.json --baseline baseline.json
        fi
"""


def format_pr_comment(results: dict[str, Any]) -> str:
    """Format results as a GitHub PR comment.

    Args:
        results: JSON results from Agentrial.

    Returns:
        Markdown-formatted comment.
    """
    summary = results.get("summary", {})
    passed = summary.get("passed", False)
    pass_rate = summary.get("overall_pass_rate", 0)
    ci = summary.get("overall_pass_rate_ci", {})

    status = "Passed" if passed else "Failed"
    status_emoji = "white_check_mark" if passed else "x"

    lines = [
        f"## Agentrial Results :{status_emoji}:",
        "",
        f"**Status**: {status}",
        f"**Overall Pass Rate**: {pass_rate:.1%} "
        f"(95% CI: {ci.get('lower', 0):.1%}-{ci.get('upper', 0):.1%})",
        f"**Total Cost**: ${summary.get('total_cost', 0):.4f}",
        "",
        "### Test Results",
        "",
        "| Test Case | Pass Rate | 95% CI | Avg Cost | Avg Latency |",
        "|-----------|-----------|--------|----------|-------------|",
    ]

    for result in results.get("results", []):
        tc = result.get("test_case", {})
        pr = result.get("pass_rate", 0)
        pr_ci = result.get("pass_rate_ci", {})
        cost = result.get("mean_cost", 0)
        latency = result.get("mean_latency_ms", 0)

        ci_str = f"{pr_ci.get('lower', 0):.0%}-{pr_ci.get('upper', 0):.0%}"
        lines.append(
            f"| {tc.get('name', 'unknown')} | {pr:.1%} | {ci_str} | ${cost:.4f} | {latency:.0f}ms |"
        )

    return "\n".join(lines)


def post_pr_comment(
    results: dict[str, Any],
    repo: str,
    pr_number: int,
    token: str,
) -> bool:
    """Post evaluation results as a PR comment.

    Requires the `requests` library.

    Args:
        results: JSON results from Agentrial.
        repo: Repository in "owner/repo" format.
        pr_number: Pull request number.
        token: GitHub token with PR comment permissions.

    Returns:
        True if comment was posted successfully.
    """
    try:
        import requests
    except ImportError as err:
        raise ImportError("requests library required for posting PR comments") from err

    comment = format_pr_comment(results)

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.post(url, headers=headers, json={"body": comment})
    return response.status_code == 201


def get_exit_code(results: dict[str, Any]) -> int:
    """Get appropriate exit code for CI based on results.

    Args:
        results: JSON results from Agentrial.

    Returns:
        0 if passed, 1 if failed.
    """
    return 0 if results.get("summary", {}).get("passed", False) else 1
