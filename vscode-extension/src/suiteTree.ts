/**
 * Tree view provider for test suites.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { AgentrialRunner } from './runner';

export class SuiteTreeProvider implements vscode.TreeDataProvider<SuiteItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<SuiteItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private suiteFiles: string[] = [];

    constructor(private runner: AgentrialRunner) {}

    refresh(): void {
        this.discoverSuites();
    }

    private async discoverSuites(): Promise<void> {
        const files = await vscode.workspace.findFiles(
            '**/*.{yml,yaml}',
            '**/node_modules/**'
        );

        // Filter for agentrial test files
        this.suiteFiles = [];
        for (const file of files) {
            try {
                const doc = await vscode.workspace.openTextDocument(file);
                const text = doc.getText(new vscode.Range(0, 0, 5, 0));
                if (text.includes('suite:') && text.includes('cases:')) {
                    this.suiteFiles.push(file.fsPath);
                }
            } catch {
                // Skip unreadable files
            }
        }

        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: SuiteItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: SuiteItem): Thenable<SuiteItem[]> {
        if (!element) {
            return Promise.resolve(
                this.suiteFiles.map(f => new SuiteItem(
                    path.basename(f),
                    f,
                    vscode.TreeItemCollapsibleState.None
                ))
            );
        }
        return Promise.resolve([]);
    }
}

class SuiteItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly filePath: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    ) {
        super(label, collapsibleState);
        this.tooltip = filePath;
        this.description = vscode.workspace.asRelativePath(filePath);
        this.contextValue = 'suite';

        this.command = {
            command: 'agentrial.runSuite',
            title: 'Run Suite',
            arguments: [filePath],
        };

        this.iconPath = new vscode.ThemeIcon('beaker');
    }
}
