import Foundation

var checks = 0
func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    checks += 1
    guard try condition() else { fatalError(message) }
}
func rejects(_ message: String, _ action: () throws -> Void) {
    checks += 1
    do { try action(); fatalError(message) } catch { /* Expected rejection. */ }
}

let server = try ServerAddress(" https://RUNE.example:443/ ")
try check(server.contains(URL(string: "https://rune.example/api/events")!), "Default HTTPS port should match")
try check(!server.contains(URL(string: "https://rune.example.evil/path")!), "Host suffix must not match")
try check(!server.contains(URL(string: "https://rune.example:444/")!), "Port must match")
try check(!server.contains(URL(string: "http://rune.example/")!), "Downgrade must not match")
try check(!server.contains(URL(string: "https://user:pass@rune.example/")!), "Embedded credentials must not match")
for raw in ["https://", "rune.example", "https://rune.example/path", "https://rune.example?token=secret", "https://rune.example#token", "https://user:pass@rune.example", "https://rune.example:65536", "https://rune.example:0", "https://rune. example", "file:///tmp/data", "http://127.0.0.1:18789"] {
    rejects("Accepted unsafe origin: \(raw)") { _ = try ServerAddress(raw) }
}
_ = try ServerAddress("http://localhost:18789", allowSimulatorHTTP: true)
_ = try ServerAddress("http://[::1]:18789", allowSimulatorHTTP: true)
rejects("Simulator exception must not allow LAN HTTP") { _ = try ServerAddress("http://192.168.1.2:18789", allowSimulatorHTTP: true) }
rejects("Simulator exception must not allow suffix spoof") { _ = try ServerAddress("http://localhost.evil", allowSimulatorHTTP: true) }
let transcript = "민수: 내일 10시에 만나요.\nAlex: sounds good 🐱"
try check(TextImport.decode(Data(transcript.utf8)) == transcript, "UTF-8 lost Unicode")
try check(TextImport.decode(transcript.data(using: .utf16)!) == transcript, "UTF-16 lost Unicode")
try check(TextImport.decode(Data([0xEF, 0xBB, 0xBF]) + Data(transcript.utf8)) == transcript, "BOM not removed")
for data in [Data(), Data("  \n".utf8), Data([0xFF, 0x00, 0x81]), Data("abc\0def".utf8), Data(repeating: 65, count: TextImport.maximumBytes + 1), Data(repeating: 65, count: 100_001)] {
    rejects("Invalid import accepted") { _ = try TextImport.decode(data) }
}
let fixture = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
try Data(transcript.utf8).write(to: fixture)
try check(TextImport.read(fixture) == transcript, "File import failed")
try FileManager.default.removeItem(at: fixture)
rejects("Missing file accepted") { _ = try TextImport.read(fixture) }
print("Passed \(checks) iOS core checks")
