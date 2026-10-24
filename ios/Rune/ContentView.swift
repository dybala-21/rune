import SwiftUI
import UniformTypeIdentifiers
import SafariServices

struct ContentView: View {
    @AppStorage("rune.serverOrigin") private var savedOrigin = ""
    @State private var address = ""
    @State private var session: RuneWebSession?
    @State private var error: String?

    var body: some View {
        Group {
            if let session {
                ConnectedView(session: session) {
                    session.disconnect()
                    self.session = nil
                }
            } else {
                NavigationStack {
                    Form {
                        Section {
                            Text("Connect to your Rune")
                                .font(.title2.bold())
                            Text("Chat and follow work from your iPhone. Your Mac or server runs the agent and must stay online.")
                                .foregroundStyle(.secondary)
                        }
                        Section("Server") {
                            TextField("https://rune.example.com", text: $address)
                                .textContentType(.URL).keyboardType(.URL)
                                .textInputAutocapitalization(.never).autocorrectionDisabled()
                                .submitLabel(.go).onSubmit(connect)
                            Button("Connect", action: connect).disabled(address.trimmingCharacters(in: .whitespaces).isEmpty)
                        }
                        Section {
                            Text("Use your authenticated HTTPS Rune gateway. Sign in on the next screen; this app saves only the server address. Login ends when you disconnect or close the app.")
                            Text("To summarize a conversation, export it as text and use Import text after connecting. Other apps' conversations are not read automatically.")
                        }.foregroundStyle(.secondary)
                        if let error { Section { Text(error).foregroundStyle(.red) } }
                    }
                    .navigationTitle("Rune")
                }
            }
        }
        .onAppear { if address.isEmpty { address = savedOrigin } }
    }

    private func connect() {
        do {
            #if DEBUG && targetEnvironment(simulator)
            let server = try ServerAddress(address, allowSimulatorHTTP: true)
            #else
            let server = try ServerAddress(address)
            #endif
            savedOrigin = server.url.absoluteString
            session = RuneWebSession(server: server)
            error = nil
        } catch { self.error = error.localizedDescription }
    }
}

private struct ConnectedView: View {
    @ObservedObject var session: RuneWebSession
    var disconnect: () -> Void
    @State private var importing = false
    @State private var draft: ImportedDraft?
    @State private var notice: String?
    @State private var showDisconnect = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Rune").font(.headline)
                Text(session.server.url.host ?? "").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                Menu {
                    Button { importing = true } label: { Label("Import text", systemImage: "doc.badge.plus") }
                    Button(action: session.openHome) { Label("Reconnect", systemImage: "arrow.clockwise") }
                    Button { showDisconnect = true } label: { Label("Disconnect", systemImage: "rectangle.portrait.and.arrow.right") }
                } label: {
                    Image(systemName: "ellipsis.circle").frame(width: 44, height: 44)
                }.accessibilityLabel("Connection options")
            }.padding(.horizontal, 12)
            if session.loading { ProgressView().frame(maxWidth: .infinity).padding(4) }
            if let error = session.error {
                VStack {
                    Text(error).font(.callout)
                    Button("Reconnect", action: session.openHome)
                }.padding().frame(maxWidth: .infinity).background(.thinMaterial)
            }
            RuneWebView(session: session)
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.plainText]) { result in
            do {
                let url = try result.get()
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                draft = ImportedDraft(name: url.lastPathComponent, text: try TextImport.read(url))
            } catch { notice = error.localizedDescription }
        }
        .sheet(item: $draft) { imported in
            ImportPreview(draft: imported, server: session.server.url.host ?? "") { text in
                try await session.insert(text)
                draft = nil
            }
        }
        .sheet(isPresented: Binding(get: { session.externalURL != nil }, set: { if !$0 { session.externalURL = nil } })) {
            if let url = session.externalURL { ExternalBrowser(url: url) }
        }
        .alert("Import text", isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })) {
            Button("OK") { notice = nil }
        } message: { Text(notice ?? "") }
        .confirmationDialog("Disconnect and discard unsent drafts?", isPresented: $showDisconnect, titleVisibility: .visible) {
            Button("Disconnect", role: .destructive, action: disconnect)
        } message: { Text("Work already submitted continues on your server.") }
    }
}

private struct ImportedDraft: Identifiable {
    let id = UUID()
    let name: String
    let text: String
}

private struct ImportPreview: View {
    let draft: ImportedDraft
    let server: String
    let insert: (String) async throws -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var error: String?
    @State private var inserting = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Review the text before adding it to the current chat. Add a summary request there, then send when ready. Sending shares it with \(server) and your selected model provider.")
                    .font(.callout).foregroundStyle(.secondary)
                ScrollView { Text(draft.text).font(.system(.body, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                if let error { Text(error).foregroundStyle(.red).font(.callout) }
                Button("Add to chat draft") {
                    inserting = true
                    Task { @MainActor in
                        defer { inserting = false }
                        do { try await insert(draft.text) }
                        catch { self.error = error.localizedDescription }
                    }
                }.buttonStyle(.borderedProminent).disabled(inserting)
            }.padding().navigationTitle(draft.name).navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
        }.interactiveDismissDisabled(inserting)
    }
}

private struct ExternalBrowser: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController { SFSafariViewController(url: url) }
    func updateUIViewController(_ uiViewController: SFSafariViewController, context: Context) {}
}
