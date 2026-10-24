import Foundation

enum TextImport {
    static let maximumBytes = 256 * 1024

    static func decode(_ data: Data) throws -> String {
        guard data.count <= maximumBytes else { throw ImportError.tooLarge }
        let text: String?
        if data.starts(with: [0xFF, 0xFE]) || data.starts(with: [0xFE, 0xFF]) {
            text = String(data: data, encoding: .utf16)
        } else {
            text = String(data: data, encoding: .utf8)
        }
        guard let text, !text.contains("\0") else { throw ImportError.encoding }
        let cleaned = text.trimmingCharacters(in: CharacterSet(charactersIn: "\u{FEFF}"))
        guard !cleaned.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ImportError.empty }
        guard cleaned.utf16.count <= 100_000 else { throw ImportError.tooLarge }
        return cleaned
    }

    static func read(_ url: URL) throws -> String {
        // Bounded read also protects against a file that grows while being imported.
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        return try decode(handle.read(upToCount: maximumBytes + 1) ?? Data())
    }
}

enum ImportError: LocalizedError {
    case tooLarge, encoding, empty
    var errorDescription: String? {
        switch self {
        case .tooLarge: return "Choose a text file under 256 KB and 100,000 characters. For larger exports, use the chat attachment button."
        case .encoding: return "Choose a UTF-8 or UTF-16 text export. This file could not be decoded."
        case .empty: return "This file contains no text."
        }
    }
}
