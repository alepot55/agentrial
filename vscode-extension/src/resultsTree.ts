/**
 * Tree view provider for test results.
 */

import * as vscode from 'vscode';
import { SuiteResult, CaseResult } from './runner';

export class ResultsTreeProvider implements vscode.TreeDataProvider<ResultItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<ResultItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private currentResult: SuiteResult | null = null;

    setResults(result: SuiteResult): void {
        this.currentResult = result;
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: ResultItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ResultItem): Thenable<ResultItem[]> {
        if (!this.currentResult) {
            return Promise.resolve([]);
        }

        if (!element) {
            // Root: show overall + each case
            const items: ResultItem[] = [
                new ResultItem(
                    `Overall: ${(this.currentResult.overall_pass_rate * 100).toFixed(0)}%`,
                    this.currentResult.passed ? 'passed' : 'failed',
                    vscode.TreeItemCollapsibleState.None
                ),
            ];

            for (const caseResult of this.currentResult.results || []) {
                const threshold = vscode.workspace.getConfiguration('agentrial').get<number>('defaultThreshold', 0.85);
                items.push(new ResultItem(
                    caseResult.name,
                    caseResult.pass_rate >= threshold ? 'passed' : 'failed',
                    vscode.TreeItemCollapsibleState.Collapsed,
                    caseResult,
                ));
            }

            return Promise.resolve(items);
        }

        // Children: case details
        if (element.caseResult) {
            const c = element.caseResult;
            return Promise.resolve([
                new ResultItem(
                    `Pass rate: ${(c.pass_rate * 100).toFixed(0)}%`,
                    'detail',
                    vscode.TreeItemCollapsibleState.None
                ),
                new ResultItem(
                    `Cost: $${c.mean_cost.toFixed(4)}`,
                    'detail',
                    vscode.TreeItemCollapsibleState.None
                ),
                new ResultItem(
                    `Latency: ${c.mean_latency_ms.toFixed(0)}ms`,
                    'detail',
                    vscode.TreeItemCollapsibleState.None
                ),
                new ResultItem(
                    `Trials: ${c.trials_count}`,
                    'detail',
                    vscode.TreeItemCollapsibleState.None
                ),
            ]);
        }

        return Promise.resolve([]);
    }
}

class ResultItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly status: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly caseResult?: CaseResult,
    ) {
        super(label, collapsibleState);

        switch (status) {
            case 'passed':
                this.iconPath = new vscode.ThemeIcon('check', new vscode.ThemeColor('testing.iconPassed'));
                break;
            case 'failed':
                this.iconPath = new vscode.ThemeIcon('x', new vscode.ThemeColor('testing.iconFailed'));
                break;
            case 'detail':
                this.iconPath = new vscode.ThemeIcon('info');
                break;
        }
    }
}
