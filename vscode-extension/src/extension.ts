/**
 * Agentrial VS Code Extension
 *
 * Thin client that invokes the agentrial CLI and displays results
 * inline in the editor, including flamegraphs and snapshot comparisons.
 */

import * as vscode from 'vscode';
import { AgentrialRunner } from './runner';
import { SuiteTreeProvider } from './suiteTree';
import { ResultsTreeProvider } from './resultsTree';
import { FlamegraphPanel } from './flamegraph';

let runner: AgentrialRunner;
let suiteTree: SuiteTreeProvider;
let resultsTree: ResultsTreeProvider;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('agentrial');
    const pythonPath = config.get<string>('pythonPath', 'python');

    runner = new AgentrialRunner(pythonPath);
    suiteTree = new SuiteTreeProvider(runner);
    resultsTree = new ResultsTreeProvider();

    // Register tree views
    vscode.window.registerTreeDataProvider('agentrial.suites', suiteTree);
    vscode.window.registerTreeDataProvider('agentrial.results', resultsTree);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('agentrial.runSuite', async (suiteFile?: string) => {
            const file = suiteFile || await selectSuiteFile();
            if (!file) return;

            const trials = config.get<number>('defaultTrials', 10);
            await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: `Running agentrial suite...`,
                    cancellable: true,
                },
                async (progress, token) => {
                    try {
                        const result = await runner.runSuite(file, trials, token);
                        resultsTree.setResults(result);
                        showResultsSummary(result);
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Agentrial: ${e.message}`);
                    }
                }
            );
        }),

        vscode.commands.registerCommand('agentrial.runTestCase', async () => {
            vscode.window.showInformationMessage(
                'Agentrial: Use gutter icons to run individual test cases'
            );
        }),

        vscode.commands.registerCommand('agentrial.showFlamegraph', async (suiteFile?: string) => {
            const file = suiteFile || await selectSuiteFile();
            if (!file) return;

            try {
                const html = await runner.getFlamegraphHtml(file);
                FlamegraphPanel.createOrShow(context.extensionUri, html);
            } catch (e: any) {
                vscode.window.showErrorMessage(`Agentrial: ${e.message}`);
            }
        }),

        vscode.commands.registerCommand('agentrial.compareSnapshot', async (suiteFile?: string) => {
            const file = suiteFile || await selectSuiteFile();
            if (!file) return;

            try {
                const comparison = await runner.compareSnapshot(file);
                showSnapshotComparison(comparison);
            } catch (e: any) {
                vscode.window.showErrorMessage(`Agentrial: ${e.message}`);
            }
        }),

        vscode.commands.registerCommand('agentrial.securityScan', async () => {
            const mcpFiles = await vscode.workspace.findFiles('**/mcp.json', '**/node_modules/**');
            if (mcpFiles.length === 0) {
                vscode.window.showWarningMessage('No mcp.json found in workspace');
                return;
            }

            const file = mcpFiles.length === 1
                ? mcpFiles[0].fsPath
                : await selectFile(mcpFiles);

            if (!file) return;

            try {
                const result = await runner.securityScan(file);
                showSecurityResults(result);
            } catch (e: any) {
                vscode.window.showErrorMessage(`Agentrial: ${e.message}`);
            }
        })
    );

    // Auto-refresh on file change
    if (config.get<boolean>('autoRefresh', true)) {
        const watcher = vscode.workspace.createFileSystemWatcher('**/*.{yml,yaml}');
        watcher.onDidChange(() => suiteTree.refresh());
        watcher.onDidCreate(() => suiteTree.refresh());
        watcher.onDidDelete(() => suiteTree.refresh());
        context.subscriptions.push(watcher);
    }

    // Discover suites on activation
    suiteTree.refresh();
}

export function deactivate() {
    // Cleanup
}

async function selectSuiteFile(): Promise<string | undefined> {
    const files = await vscode.workspace.findFiles(
        '**/*.{yml,yaml}',
        '**/node_modules/**'
    );

    const suiteFiles = files.filter(f =>
        f.fsPath.includes('agentrial') || f.fsPath.includes('test')
    );

    if (suiteFiles.length === 0) {
        vscode.window.showWarningMessage('No agentrial test files found');
        return undefined;
    }

    if (suiteFiles.length === 1) {
        return suiteFiles[0].fsPath;
    }

    const pick = await vscode.window.showQuickPick(
        suiteFiles.map(f => ({
            label: vscode.workspace.asRelativePath(f),
            detail: f.fsPath,
        })),
        { placeHolder: 'Select a test suite file' }
    );

    return pick?.detail;
}

async function selectFile(files: vscode.Uri[]): Promise<string | undefined> {
    const pick = await vscode.window.showQuickPick(
        files.map(f => ({
            label: vscode.workspace.asRelativePath(f),
            detail: f.fsPath,
        })),
        { placeHolder: 'Select a file' }
    );
    return pick?.detail;
}

function showResultsSummary(result: any) {
    const passed = result.passed;
    const passRate = (result.overall_pass_rate * 100).toFixed(0);
    const icon = passed ? '$(check)' : '$(x)';

    vscode.window.showInformationMessage(
        `${icon} Agentrial: ${passRate}% pass rate ` +
        `(${result.results?.length || 0} cases, ` +
        `cost: $${result.total_cost?.toFixed(4) || '0.00'})`
    );
}

function showSnapshotComparison(comparison: any) {
    const regressions = comparison.regressions || [];
    if (regressions.length === 0) {
        vscode.window.showInformationMessage(
            '$(check) Agentrial: No regressions detected'
        );
    } else {
        vscode.window.showWarningMessage(
            `$(warning) Agentrial: ${regressions.length} regression(s) detected`
        );
    }
}

function showSecurityResults(result: any) {
    const score = result.score?.toFixed(1) || '?';
    const findings = result.findings?.length || 0;
    const critical = result.critical_count || 0;

    if (critical > 0) {
        vscode.window.showWarningMessage(
            `$(shield) MCP Security: ${score}/10 — ${critical} critical finding(s)`
        );
    } else if (findings > 0) {
        vscode.window.showInformationMessage(
            `$(shield) MCP Security: ${score}/10 — ${findings} finding(s)`
        );
    } else {
        vscode.window.showInformationMessage(
            `$(shield) MCP Security: 10/10 — No issues found`
        );
    }
}
