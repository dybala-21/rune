import Darwin
import Dispatch
import Foundation

@MainActor
final class CommandOutput {
    private let output: FileHandle
    private var pending: Pending?
    private var failed = false

    private final class Pending {
        let id: UUID
        let data: Data
        let check: @MainActor () throws -> Void
        let continuation: CheckedContinuation<Void, Error>
        let source: DispatchSourceWrite
        let timer: DispatchSourceTimer
        var offset = 0

        init(id: UUID, data: Data, check: @escaping @MainActor () throws -> Void,
             continuation: CheckedContinuation<Void, Error>, fd: Int32) {
            self.id = id
            self.data = data
            self.check = check
            self.continuation = continuation
            source = DispatchSource.makeWriteSource(fileDescriptor: fd, queue: .main)
            timer = DispatchSource.makeTimerSource(queue: .main)
        }
    }

    init(_ output: FileHandle) throws {
        let fd = output.fileDescriptor
        let flags = fcntl(fd, F_GETFL)
        guard flags >= 0, fcntl(fd, F_SETFL, flags | O_NONBLOCK) != -1,
              fcntl(fd, F_SETNOSIGPIPE, 1) != -1 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        self.output = output
    }

    func write(_ data: Data, check: @escaping @MainActor () throws -> Void = {}) async throws {
        guard !failed, pending == nil else { throw CocoaError(.fileWriteUnknown) }
        guard data.count <= 32 * 1024 * 1024 else { throw CocoaError(.fileWriteOutOfSpace) }
        try Task.checkCancellation()
        try check()
        if data.isEmpty { return }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let request = Pending(id: id, data: data, check: check,
                                      continuation: continuation, fd: output.fileDescriptor)
                pending = request
                request.source.setEventHandler { [self] in
                    MainActor.assumeIsolated { drain(id) }
                }
                // FIFOs may miss a write event when the reader disconnects.
                request.timer.schedule(deadline: .now(), repeating: .milliseconds(100))
                request.timer.setEventHandler { [self] in
                    MainActor.assumeIsolated { drain(id) }
                }
                request.source.resume()
                request.timer.resume()
            }
        } onCancel: {
            Task { @MainActor in self.finish(id, error: CancellationError()) }
        }
    }

    private func drain(_ id: UUID) {
        guard let request = pending, request.id == id else { return }
        do {
            try request.check()
            let count = request.data.withUnsafeBytes { bytes in
                Darwin.write(output.fileDescriptor, bytes.baseAddress!.advanced(by: request.offset),
                             min(64 * 1024, bytes.count - request.offset))
            }
            if count > 0 {
                request.offset += count
                if request.offset == request.data.count { finish(id) }
            } else if count == 0 {
                throw POSIXError(.EIO)
            } else if errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        } catch {
            finish(id, error: error)
        }
    }

    private func finish(_ id: UUID, error: Error? = nil) {
        guard let request = pending, request.id == id else { return }
        pending = nil
        request.source.cancel()
        request.timer.cancel()
        if let error {
            // A partial JSON response cannot be followed by another response.
            failed = true
            request.continuation.resume(throwing: error)
        } else {
            request.continuation.resume()
        }
    }
}
