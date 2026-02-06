/**
 * Runner that invokes the agentrial CLI and parses JSON output.
 */

import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface SuiteResult {
    passed: boolean;
    overall_pass_rate: number;
    total_cost: number;
    total_duration_ms: number;
    results: CaseResult[];
}

export interface CaseResult {
    name: string;
    pass_rate: number;
    mean_cost: number;
    mean_latency_ms: number;
    trials_count: number;
}

export interface SecurityResult {
    score: number;
    findings: SecurityFinding[];
    tools_scanned: number;
    servers_scanned: number;
    critical_count: number;
    high_count: number;
    passed: boolean;
}

export interface SecurityFinding {
    severity: string;
    category: string;
    title: string;
    description: string;
    tool_name?: string;
    server_name?: string;
    recommendation: string;
}

export class AgentrialRunner {
    private pythonPath: string;

    constructor(pythonPath: string = 'python') {
        this.pythonPath = pythonPath;
    }

    /**
     * Run a test suite and return parsed results.
     */
    async runSuite(
        suiteFile: string,
        trials?: number,
        token?: vscode.CancellationToken,
    ): Promise<SuiteResult> {
        let cmd = `${this.pythonPath} -m agentrial run "${suiteFile}" --json`;
        if (trials) {
            cmd += ` --trials ${trials}`;
        }

        const result = await this.execute(cmd, token);
        return JSON.parse(result);
    }

    /**
     * Get flamegraph HTML for a suite.
     */
    async getFlamegraphHtml(suiteFile: string): Promise<string> {
        const cmd = `${this.pythonPath} -m agentrial run "${suiteFile}" --flamegraph --html -`;
        return await this.execute(cmd);
    }

    /**
     * Compare current results with snapshot.
     */
    async compareSnapshot(suiteFile: string): Promise<any> {
        const cmd = `${this.pythonPath} -m agentrial run "${suiteFile}" --json --compare-snapshot`;
        const result = await this.execute(cmd);
        return JSON.parse(result);
    }

    /**
     * Run MCP security scan.
     */
    async securityScan(mcpConfigPath: string): Promise<SecurityResult> {
        const cmd = `${this.pythonPath} -m agentrial security scan "${mcpConfigPath}" --json`;
        const result = await this.execute(cmd);
        return JSON.parse(result);
    }

    /**
     * Discover test suite files in the workspace.
     */
    async discoverSuites(): Promise<string[]> {
        const cmd = `${this.pythonPath} -m agentrial list --json`;
        try {
            const result = await this.execute(cmd);
            const data = JSON.parse(result);
            return data.suites || [];
        } catch {
            return [];
        }
    }

    /**
     * Check if agentrial is installed and accessible.
     */
    async checkInstallation(): Promise<boolean> {
        try {
            await this.execute(`${this.pythonPath} -m agentrial --version`);
            return true;
        } catch {
            return false;
        }
    }

    private async execute(
        cmd: string,
        token?: vscode.CancellationToken,
    ): Promise<string> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

        return new Promise((resolve, reject) => {
            const proc = exec(
                cmd,
                {
                    cwd: workspaceFolder,
                    maxBuffer: 10 * 1024 * 1024, // 10 MB
                    timeout: 300000, // 5 minutes
                },
                (error, stdout, stderr) => {
                    if (error) {
                        reject(new Error(stderr || error.message));
                    } else {
                        resolve(stdout);
                    }
                }
            );

            // Handle cancellation
            if (token) {
                token.onCancellationRequested(() => {
                    proc.kill('SIGTERM');
                    reject(new Error('Cancelled by user'));
                });
            }
        });
    }
}
