import SwiftUI
import WebKit

@MainActor
final class RuneWebSession: NSObject, ObservableObject, WKNavigationDelegate, WKUIDelegate {
    let server: ServerAddress
    let webView: WKWebView
    @Published var error: String?
    @Published var loading = true
    @Published var externalURL: URL?

    init(server: ServerAddress) {
        self.server = server
        let configuration = WKWebViewConfiguration()
        // Cookies stay in this connection, never in Safari or UserDefaults.
        configuration.websiteDataStore = .nonPersistent()
        self.webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsLinkPreview = false
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        openHome()
    }

    func openHome() {
        error = nil
        loading = true
        // Always GET the root: never replay a POST from the login/history stack.
        webView.load(URLRequest(url: server.url))
    }

    func disconnect() {
        webView.stopLoading()
        webView.navigationDelegate = nil
        webView.uiDelegate = nil
    }

    func insert(_ text: String) async throws {
        guard let current = webView.url, server.contains(current), !loading else {
            throw ConnectionError.importUnavailable
        }
        // Pass data as arguments, not interpolated executable JavaScript.
        let accepted = try await webView.callAsyncJavaScript(
            "return location.origin === new URL(origin).origin && typeof window.runeImportText === 'function' && window.runeImportText(text);",
            arguments: ["origin": server.url.absoluteString, "text": text],
            in: nil, contentWorld: .page
        )
        guard accepted as? Bool == true else { throw ConnectionError.importUnavailable }
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { decisionHandler(.cancel); return }
        if server.contains(url) {
            if action.targetFrame == nil {
                decisionHandler(.cancel)
                // New windows are only supported for explicit GET links.
                if action.navigationType == .linkActivated && action.request.httpMethod == "GET" {
                    webView.load(action.request)
                }
            } else {
                decisionHandler(.allow)
            }
            return
        }
        decisionHandler(.cancel)
        if action.navigationType == .linkActivated,
           ["https", "http"].contains(url.scheme?.lowercased() ?? ""),
           url.user == nil, url.password == nil {
            externalURL = url
        } else if action.targetFrame?.isMainFrame != false {
            error = "Navigation outside your Rune server was blocked. Open external services using their links."
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        loading = true
        error = nil
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loading = false }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        report(error)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { report(error) }

    private func report(_ error: Error) {
        loading = false
        if (error as NSError).code != NSURLErrorCancelled { self.error = error.localizedDescription }
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        loading = false
        error = "The page stopped. Reconnect to restore saved work. Unsent drafts may need to be entered again."
    }

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? { nil }
}

struct RuneWebView: UIViewRepresentable {
    @ObservedObject var session: RuneWebSession
    func makeUIView(context: Context) -> WKWebView { session.webView }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
}
