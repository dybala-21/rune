import CryptoKit
import Foundation

enum ObservedText {
    static func fingerprint(_ text: String) -> String {
        let data = Data(text.precomposedStringWithCanonicalMapping.utf8)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func fields(_ key: String, _ text: String?) -> [String: Any] {
        guard let text else { return [:] }
        let normalized = text.precomposedStringWithCanonicalMapping
        let prefix = String(normalized.prefix(500))
        var result: [String: Any] = [key: prefix]
        if prefix != normalized {
            result[key + "Truncated"] = true
            result[key + "SHA256"] = fingerprint(normalized)
        }
        return result
    }
}
