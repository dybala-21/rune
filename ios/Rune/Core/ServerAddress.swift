import Foundation

struct ServerAddress: Equatable {
    let url: URL

    init(_ input: String, allowSimulatorHTTP: Bool = false) throws {
        let value = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.contains(where: { $0.isWhitespace }),
              var parts = URLComponents(string: value),
              let host = parts.host, !host.isEmpty,
              parts.user == nil, parts.password == nil,
              parts.query == nil, parts.fragment == nil,
              parts.path.isEmpty || parts.path == "/",
              parts.port.map({ (1...65535).contains($0) }) ?? true else {
            throw ConnectionError.invalidAddress
        }
        let local = ["localhost", "127.0.0.1", "[::1]", "::1"].contains(host.lowercased())
        guard parts.scheme?.lowercased() == "https" ||
              (allowSimulatorHTTP && local && parts.scheme?.lowercased() == "http") else {
            throw ConnectionError.httpsRequired
        }
        parts.scheme = parts.scheme?.lowercased()
        parts.host = host.lowercased()
        parts.path = "/"
        guard let url = parts.url else { throw ConnectionError.invalidAddress }
        self.url = url
    }

    func contains(_ candidate: URL) -> Bool {
        guard candidate.user == nil, candidate.password == nil else { return false }
        func port(_ url: URL) -> Int { url.port ?? (url.scheme == "https" ? 443 : 80) }
        return candidate.scheme?.lowercased() == url.scheme &&
            candidate.host?.lowercased() == url.host?.lowercased() && port(candidate) == port(url)
    }
}

enum ConnectionError: LocalizedError {
    case invalidAddress, httpsRequired, importUnavailable
    var errorDescription: String? {
        switch self {
        case .invalidAddress: return "Enter the server origin only, for example https://rune.example.com. Do not include a path, password, or token."
        case .httpsRequired: return "Use an HTTPS Rune server. HTTP localhost is available only in a Debug simulator build."
        case .importUnavailable: return "Open a connected chat first. The text was not inserted; check that the combined draft is under 100,000 characters."
        }
    }
}
