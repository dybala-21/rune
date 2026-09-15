import Foundation

enum CommandInput {
    @MainActor static func consume(_ input: FileHandle, handle: (Data) async throws -> Void) async throws {
        var command = Data()
        for try await byte in input.bytes {
            if byte == 10 {
                try await handle(command)
                command.removeAll(keepingCapacity: true)
            } else {
                guard command.count < 32768 else {
                    throw CocoaError(.fileReadTooLarge)
                }
                command.append(byte)
            }
        }
        guard command.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
    }
}
