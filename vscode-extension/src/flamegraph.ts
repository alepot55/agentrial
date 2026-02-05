/**
 * Flamegraph webview panel for displaying trajectory analysis.
 */

import * as vscode from 'vscode';

export class FlamegraphPanel {
    public static currentPanel: FlamegraphPanel | undefined;
    private static readonly viewType = 'agentrial.flamegraph';

    private readonly panel: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];

    private constructor(
        panel: vscode.WebviewPanel,
        extensionUri: vscode.Uri,
        html: string,
    ) {
        this.panel = panel;
        this.panel.webview.html = this.getHtmlContent(html);

        this.panel.onDidDispose(
            () => this.dispose(),
            null,
            this.disposables
        );
    }

    public static createOrShow(
        extensionUri: vscode.Uri,
        html: string,
    ): void {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        // If we already have a panel, update it
        if (FlamegraphPanel.currentPanel) {
            FlamegraphPanel.currentPanel.panel.reveal(column);
            FlamegraphPanel.currentPanel.update(html);
            return;
        }

        // Create a new panel
        const panel = vscode.window.createWebviewPanel(
            FlamegraphPanel.viewType,
            'Agentrial Flamegraph',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            }
        );

        FlamegraphPanel.currentPanel = new FlamegraphPanel(
            panel,
            extensionUri,
            html,
        );
    }

    public update(html: string): void {
        this.panel.webview.html = this.getHtmlContent(html);
    }

    private getHtmlContent(agentrialHtml: string): string {
        // The CLI generates a complete standalone HTML page,
        // so we can use it directly in the webview
        return agentrialHtml;
    }

    public dispose(): void {
        FlamegraphPanel.currentPanel = undefined;
        this.panel.dispose();
        while (this.disposables.length) {
            const d = this.disposables.pop();
            if (d) {
                d.dispose();
            }
        }
    }
}
