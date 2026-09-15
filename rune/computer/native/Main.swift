import AppKit
import CoreServices
import Foundation

@main
struct RuneComputer {
    @MainActor static func main() {
        NSApplication.shared.setActivationPolicy(.accessory)
        guard CommandLine.arguments.count == 2 else { return }
        let path = CommandLine.arguments[1]
        let fd = open(path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW)
        var info = stat()
        guard fd > 2, fstat(fd, &info) == 0, info.st_uid == geteuid(),
              info.st_mode & S_IFMT == S_IFIFO, info.st_mode & 0o777 == 0o600,
              let control = try? HostControl(fd: fd) else { return }
        control.watchDisconnect()
        guard LSRegisterURL(Bundle.main.bundleURL as CFURL, true) == noErr else {
            FileHandle.standardError.write(Data("Could not register Rune Computer with Launch Services\n".utf8))
            return
        }
        Task { @MainActor in
            await serve(control)
            NSApplication.shared.terminate(nil)
        }
        NSApplication.shared.run()
    }

    @MainActor static func serve(_ control: HostControl) async {
        let host = DesktopHost(control: control)
        let hello: [String: Any] = ["protocol": 3, "bundleId": Bundle.main.bundleIdentifier ?? "",
                                  "appPath": Bundle.main.bundlePath, "pid": getpid()]
        guard let helloData = try? JSONSerialization.data(withJSONObject: hello) else { return }
        do {
            let output = try CommandOutput(.standardOutput)
            try await output.write(helloData + Data([10])) { try control.check(0) }
            try await CommandInput.consume(FileHandle.standardInput) { data in
                try await respond(data, host: host, control: control, output: output)
            }
        } catch {
            FileHandle.standardError.write(Data("Native command transport failed: \(error)\n".utf8))
        }
    }

    @MainActor static func respond(_ data: Data, host: DesktopHost, control: HostControl,
                                   output: CommandOutput) async throws {
        var id = ""
        var reply: [String: Any]
        do {
            guard let request = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let requestID = request["id"] as? String,
                  let epoch = request["epoch"] as? Int,
                  let method = request["method"] as? String,
                  let params = request["params"] as? [String: Any] else {
                throw HostError(message: "Invalid native request")
            }
            id = requestID
            host.epoch = epoch
            try control.check(epoch)
            let result = try await host.handle(method, params)
            reply = ["id": id, "ok": true, "data": result]
        } catch let error as HostError {
            reply = ["id": id, "ok": false, "error": error.message, "outcome": error.outcome]
        } catch {
            reply = ["id": id, "ok": false, "error": "Native operation failed: \(error)", "outcome": "unknown"]
        }
        let data = try JSONSerialization.data(withJSONObject: reply, options: [.sortedKeys])
        let epoch = host.epoch
        try await output.write(data + Data([10])) { try control.check(epoch) }
    }
}
