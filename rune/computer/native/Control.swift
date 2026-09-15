import Darwin
import Dispatch
import Foundation

final class HostControl: @unchecked Sendable {
    private let fd: Int32
    private let lock = NSLock()
    private var epoch = 0
    private var disconnected = false
    private var monitor: DispatchSourceTimer?

    init(fd: Int32) throws {
        guard fd > 2, fcntl(fd, F_SETFL, O_NONBLOCK) != -1 else {
            throw HostError(message: "Missing private control pipe")
        }
        self.fd = fd
    }

    private func drain() {
        var bytes = [UInt8](repeating: 0, count: 256)
        while true {
            let count = read(fd, &bytes, bytes.count)
            if count > 0 {
                epoch += count
            } else if count < 0 && errno == EINTR {
                continue
            } else if count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) {
                break
            } else {
                disconnected = true
                break
            }
        }
    }

    func check(_ expected: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        drain()
        guard !disconnected else {
            throw HostError(message: "Desktop control disconnected. No further input is allowed.")
        }
        guard expected == epoch else {
            throw HostError(message: "Desktop control changed. Observe again before requesting input.")
        }
    }

    func watchDisconnect() {
        let source = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        // Named pipes can reach EOF without producing a read event on macOS.
        source.schedule(deadline: .now(), repeating: .milliseconds(100), leeway: .milliseconds(10))
        source.setEventHandler { [self] in
            lock.lock()
            drain()
            let lost = disconnected
            lock.unlock()
            if lost { _exit(0) }
        }
        monitor = source
        source.resume()
    }
}
