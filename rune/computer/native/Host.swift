import AppKit
import ApplicationServices
import CryptoKit
import ScreenCaptureKit

struct HostError: Error {
    let message: String
    var outcome = "not_executed"
    var code = "native_error"
}

struct Element {
    let handle: AXUIElement
    let signature: String
    let actions: [String]
    let writable: Bool
}

struct Observation {
    let id: String
    let app: String
    let pid: pid_t
    let window: CGWindowID
    let root: AXUIElement
    let bounds: CGRect
    let width: Int
    let height: Int
    let digest: String
    let elements: [String: Element]
    let captured: TimeInterval
}

@MainActor
final class DesktopHost {
    let control: HostControl
    var epoch = 0
    var allowed = Set<String>()
    var expires: TimeInterval = 0
    var observation: Observation?
    var uncertain = false

    init(control: HostControl) {
        self.control = control
    }

    func review(_ title: String, detail: String) -> Bool {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = title
        alert.informativeText = "Rune Computer needs your approval. This applies only to the operation shown below."
        alert.addButton(withTitle: "Allow once").keyEquivalent = ""
        alert.addButton(withTitle: "Cancel").keyEquivalent = "\r"
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 440, height: 150))
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        let text = NSTextView(frame: NSRect(x: 0, y: 0, width: 420, height: 150))
        text.isEditable = false
        text.isSelectable = true
        text.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        text.string = detail
        scroll.documentView = text
        alert.accessoryView = scroll
        let timer = Timer(timeInterval: 0.1, repeats: true) { [self] _ in
            MainActor.assumeIsolated {
                do { try control.check(epoch) }
                catch {
                    NSApplication.shared.abortModal()
                    alert.window.orderOut(nil)
                }
            }
        }
        RunLoop.main.add(timer, forMode: .modalPanel)
        defer { timer.invalidate() }
        NSApplication.shared.activate()
        return alert.runModal() == .alertFirstButtonReturn
    }

    func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
        return value
    }

    func fullString(_ element: AXUIElement, _ name: String) -> String? {
        let value = attribute(element, name)
        if let text = value as? String { return text }
        if let number = value as? NSNumber { return number.stringValue }
        return nil
    }

    func string(_ element: AXUIElement, _ name: String) -> String {
        String((fullString(element, name) ?? "").prefix(500))
    }

    func rect(_ element: AXUIElement) -> CGRect? {
        guard let p = attribute(element, kAXPositionAttribute), CFGetTypeID(p) == AXValueGetTypeID(),
              let s = attribute(element, kAXSizeAttribute), CFGetTypeID(s) == AXValueGetTypeID() else { return nil }
        var point = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(p as! AXValue, .cgPoint, &point),
              AXValueGetValue(s as! AXValue, .cgSize, &size) else { return nil }
        return CGRect(origin: point, size: size)
    }

    func signature(_ element: AXUIElement) -> String {
        let names = [kAXRoleAttribute, kAXSubroleAttribute, kAXTitleAttribute, kAXDescriptionAttribute,
                     kAXValueAttribute, kAXEnabledAttribute]
        let values = names.map { ObservedText.fingerprint(fullString(element, $0) ?? "") }
        return values.joined(separator: "\u{1f}") + (rect(element).map { NSStringFromRect($0) } ?? "")
    }

    func applications() -> [String: URL] {
        var result: [String: URL] = [:]
        let roots = ["/Applications", "/System/Applications", "/System/Applications/Utilities",
                     FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications").path]
        for root in roots {
            guard let urls = try? FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: root),
                        includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { continue }
            for url in urls where url.pathExtension == "app" {
                if let id = Bundle(url: url)?.bundleIdentifier { result[id] = url }
            }
        }
        for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
            if let id = app.bundleIdentifier, let url = app.bundleURL { result[id] = url }
        }
        // The host must never approve or configure itself through its input tools.
        result.removeValue(forKey: "dev.rune.computer")
        result.removeValue(forKey: "dev.rune.desktop")
        return result
    }

    func check(_ app: String) throws {
        try control.check(epoch)
        guard allowed.contains(app), ProcessInfo.processInfo.systemUptime < expires else {
            throw HostError(message: "This app has no current desktop grant.")
        }
        guard AXIsProcessTrusted(), CGPreflightScreenCaptureAccess() else {
            throw HostError(message: "Rune Computer needs Accessibility and Screen Recording permissions.")
        }
    }

    func running(_ app: String) throws -> NSRunningApplication {
        try check(app)
        let matches = NSRunningApplication.runningApplications(withBundleIdentifier: app)
        guard matches.count == 1, let current = matches.first, !current.isTerminated else {
            throw HostError(message: "The app is not running or its identity is ambiguous. Open it again.")
        }
        return current
    }

    func focusedWindow(_ pid: pid_t) throws -> AXUIElement {
        let root = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(root, 1)
        guard let window = attribute(root, kAXFocusedWindowAttribute), CFGetTypeID(window) == AXUIElementGetTypeID() else {
            throw HostError(message: "The app has no accessible focused window. Select a window and try again.", code: "window_unavailable")
        }
        return window as! AXUIElement
    }

    func capture(_ window: SCWindow) async throws -> (String, String, Int, Int) {
        let config = SCStreamConfiguration()
        let scale = min(1.0, 1600.0 / max(window.frame.width, window.frame.height))
        config.width = max(1, Int(window.frame.width * scale))
        config.height = max(1, Int(window.frame.height * scale))
        config.showsCursor = false
        config.ignoreShadowsSingleWindow = true
        let image = try await SCScreenshotManager.captureImage(
            contentFilter: SCContentFilter(desktopIndependentWindow: window), configuration: config)
        let bitmap = NSBitmapImageRep(cgImage: image)
        guard let png = bitmap.representation(using: .png, properties: [:]), png.count <= 5 * 1024 * 1024 else {
            throw HostError(message: "The app screenshot exceeds the size limit.")
        }
        let digest = SHA256.hash(data: png).map { String(format: "%02x", $0) }.joined()
        return (png.base64EncodedString(), digest, image.width, image.height)
    }

    func walk(_ root: AXUIElement, bounds: CGRect, scaleX: Double, scaleY: Double) throws -> ([String: Element], [[String: Any]]) {
        var queue: [(AXUIElement, Int)] = [(root, 0)]
        var index = 0
        var elements: [String: Element] = [:]
        var rows: [[String: Any]] = []
        while index < queue.count && index < 2000 {
            let (element, depth) = queue[index]
            index += 1
            if string(element, kAXSubroleAttribute) == kAXSecureTextFieldSubrole {
                throw HostError(message: "This window contains a secure input. Complete authentication yourself, then observe again.")
            }
            if rows.count < 200 {
                var actionNames: CFArray?
                AXUIElementCopyActionNames(element, &actionNames)
                let actions = actionNames as? [String] ?? []
                var writable = DarwinBoolean(false)
                AXUIElementIsAttributeSettable(element, kAXValueAttribute as CFString, &writable)
                let role = string(element, kAXRoleAttribute)
                let names = [fullString(element, kAXTitleAttribute), fullString(element, kAXDescriptionAttribute)].compactMap { $0 }
                let name = names.first(where: { !$0.isEmpty }) ?? names.first
                let value = fullString(element, kAXValueAttribute)
                if !(name ?? "").isEmpty || !(value ?? "").isEmpty || !actions.isEmpty || writable.boolValue {
                    let ref = UUID().uuidString
                    elements[ref] = Element(handle: element, signature: signature(element), actions: actions, writable: writable.boolValue)
                    var row: [String: Any] = ["ref": ref, "role": role,
                                             "press": actions.contains(kAXPressAction), "writable": writable.boolValue]
                    row.merge(ObservedText.fields("name", name)) { _, new in new }
                    row.merge(ObservedText.fields("value", value)) { _, new in new }
                    if let frame = rect(element) {
                        row["bounds"] = ["x": (frame.minX - bounds.minX) * scaleX, "y": (frame.minY - bounds.minY) * scaleY,
                                         "width": frame.width * scaleX, "height": frame.height * scaleY]
                    }
                    rows.append(row)
                }
            }
            if depth < 16, let children = attribute(element, kAXChildrenAttribute) as? [AXUIElement] {
                queue.append(contentsOf: children.prefix(2000 - min(queue.count, 2000)).map { ($0, depth + 1) })
            }
        }
        if index >= 2000 || queue.contains(where: { $0.1 == 16 }) {
            throw HostError(message: "The app's accessibility tree exceeds the inspection limit. Focus a smaller window.")
        }
        return (elements, rows)
    }

    func observe(_ app: String) async throws -> [String: Any] {
        let delays = [100, 150, 250, 400, 600]
        for attempt in 0...delays.count {
            do { return try await observeWindow(app) }
            catch let error as HostError where error.code == "window_unavailable" && attempt < delays.count {
                try check(app)
                try await Task.sleep(for: .milliseconds(delays[attempt]))
            }
        }
        throw HostError(message: "The app window did not settle. Select its window and observe again.")
    }

    func observeWindow(_ app: String) async throws -> [String: Any] {
        observation = nil
        let process = try running(app)
        let root = try focusedWindow(process.processIdentifier)
        guard let accessibleBounds = rect(root) else {
            throw HostError(message: "Cannot establish window bounds.", code: "window_unavailable")
        }
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        let matches = content.windows.filter {
            $0.owningApplication?.processID == process.processIdentifier && WindowGeometry.matches($0.frame, accessibleBounds)
        }
        guard matches.count == 1, let window = matches.first else {
            throw HostError(message: "Cannot match the accessible window to a unique visible window.", code: "window_unavailable")
        }
        let bounds = window.frame
        // Inspect secure inputs before capturing or returning pixels.
        let (elements, rows) = try walk(root, bounds: bounds, scaleX: 1, scaleY: 1)
        let (image, digest, width, height) = try await capture(window)
        try check(app)
        guard let currentBounds = rect(root), WindowGeometry.matches(currentBounds, bounds), !process.isTerminated,
              CFEqual(root, try focusedWindow(process.processIdentifier)) else {
            throw HostError(message: "The window changed while capturing it. Observe again.", code: "window_unavailable")
        }
        let scaled = rows.map { row -> [String: Any] in
            var row = row
            if let box = row["bounds"] as? [String: Double] {
                row["bounds"] = ["x": box["x"]! * Double(width) / bounds.width,
                                 "y": box["y"]! * Double(height) / bounds.height,
                                 "width": box["width"]! * Double(width) / bounds.width,
                                 "height": box["height"]! * Double(height) / bounds.height]
            }
            return row
        }
        let id = UUID().uuidString
        observation = Observation(id: id, app: app, pid: process.processIdentifier, window: window.windowID, root: root,
            bounds: bounds, width: width, height: height, digest: digest, elements: elements,
            captured: ProcessInfo.processInfo.systemUptime)
        return ["app": app, "observation": id, "title": window.title ?? "", "width": width, "height": height,
                "controls": scaled, "image_base64": image, "capturedAt": Int(Date().timeIntervalSince1970 * 1000)]
    }

    func point(_ params: [String: Any], _ x: String, _ y: String, _ view: Observation) throws -> CGPoint {
        guard let px = params[x] as? Double, let py = params[y] as? Double,
              px.isFinite, py.isFinite, px >= 0, py >= 0, px < Double(view.width), py < Double(view.height) else {
            throw HostError(message: "Coordinates must be inside the observed window screenshot.")
        }
        return CGPoint(x: view.bounds.minX + px * view.bounds.width / Double(view.width),
                       y: view.bounds.minY + py * view.bounds.height / Double(view.height))
    }

    func mouse(_ type: CGEventType, at point: CGPoint) throws {
        guard let event = CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: point, mouseButton: .left) else {
            throw HostError(message: "Could not construct mouse input.", outcome: "unknown")
        }
        event.post(tap: .cghidEventTap)
    }

    func checkHit(_ point: CGPoint, pid: pid_t) throws {
        var element: AXUIElement?
        guard AXUIElementCopyElementAtPosition(AXUIElementCreateSystemWide(), Float(point.x), Float(point.y), &element) == .success,
              let element else { throw HostError(message: "Cannot verify the app under the input position.") }
        var actualPID: pid_t = 0
        guard AXUIElementGetPid(element, &actualPID) == .success, actualPID == pid else {
            throw HostError(message: "Another app covers the input target. Observe again.")
        }
    }

    func key(_ code: CGKeyCode, flags: CGEventFlags = [], text: String? = nil) throws {
        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: true),
              let up = CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: false) else {
            throw HostError(message: "Could not construct keyboard input.", outcome: "unknown")
        }
        down.flags = flags
        up.flags = flags
        if let text = text {
            let chars = Array(text.utf16)
            chars.withUnsafeBufferPointer { buffer in
                down.keyboardSetUnicodeString(stringLength: chars.count, unicodeString: buffer.baseAddress)
                up.keyboardSetUnicodeString(stringLength: chars.count, unicodeString: buffer.baseAddress)
            }
        }
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
    }

    func act(_ params: [String: Any]) async throws -> [String: Any] {
        guard !uncertain, let view = observation, params["observation"] as? String == view.id,
              ProcessInfo.processInfo.systemUptime - view.captured < 150 else {
            throw HostError(message: "The observation expired, was consumed, or an earlier action is uncertain.")
        }
        observation = nil
        if params["action"] as? String == "publish" {
            return try await publish(view)
        }
        let process = try running(view.app)
        guard process.processIdentifier == view.pid else { throw HostError(message: "The app restarted. Observe again.") }
        let summary = try JSONSerialization.data(withJSONObject: params, options: [.prettyPrinted, .sortedKeys])
        let target = view.elements[params["ref"] as? String ?? ""]
        let name = target.map { string($0.handle, kAXTitleAttribute) + " " + string($0.handle, kAXDescriptionAttribute) } ?? "Window coordinates or keyboard focus"
        guard review("Allow one input in \(process.localizedName ?? view.app)?",
                     detail: "App: \(view.app)\nWindow: \(view.window)\nControl: \(name)\n\n" + String(decoding: summary, as: UTF8.self)) else {
            throw HostError(message: "The native input was declined. No input was sent.")
        }
        guard ProcessInfo.processInfo.systemUptime - view.captured < 150 else {
            throw HostError(message: "The observation expired during review. Observe again.")
        }
        try check(view.app)
        // Approval happens in Rune, so restore the approved app before checking its target.
        guard process.activate(options: []) else { throw HostError(message: "Could not activate the approved app.") }
        try await Task.sleep(for: .milliseconds(150))
        let root = try focusedWindow(view.pid)
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == view.pid, CFEqual(root, view.root),
              let bounds = rect(root), WindowGeometry.matches(bounds, view.bounds) else {
            throw HostError(message: "The foreground app or its window changed. Observe again.")
        }
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        guard let window = content.windows.first(where: { $0.windowID == view.window && $0.owningApplication?.processID == view.pid && WindowGeometry.matches($0.frame, view.bounds) }) else {
            throw HostError(message: "The approved window is no longer visible.")
        }
        let (_, digest, _, _) = try await capture(window)
        _ = try walk(root, bounds: view.bounds, scaleX: 1, scaleY: 1)
        guard let action = params["action"] as? String else { throw HostError(message: "Missing action") }
        let ref = params["ref"] as? String ?? ""
        let element = view.elements[ref]
        if action == "press" || action == "set_value" {
            guard let element, element.signature == signature(element.handle),
                  string(element.handle, kAXEnabledAttribute) != "0" else {
                throw HostError(message: "The observed control changed or is disabled. Observe again.")
            }
            guard let elementWindow = attribute(element.handle, kAXWindowAttribute), CFEqual(elementWindow, root) else {
                throw HostError(message: "The control no longer belongs to the approved window.")
            }
            if action == "press" && !element.actions.contains(kAXPressAction) { throw HostError(message: "This control has no press action.") }
            if action == "set_value" && !element.writable { throw HostError(message: "This control is not editable.") }
        } else if digest != view.digest {
            throw HostError(message: "The screen changed since it was observed. Read it again before coordinate or keyboard input.")
        }
        let start: CGPoint? = ["click", "drag"].contains(action) ? try point(params, "x", "y", view) : nil
        let end: CGPoint? = action == "drag" ? try point(params, "endX", "endY", view) : nil
        let text = params["text"] as? String ?? ""
        guard text.count <= 2000 else { throw HostError(message: "Input is too long.") }
        let keys: [String: CGKeyCode] = ["return": 36, "tab": 48, "escape": 53, "space": 49, "backspace": 51,
            "up": 126, "down": 125, "left": 123, "right": 124, "a": 0, "c": 8, "v": 9, "s": 1, "n": 45, "o": 31, "w": 13, "z": 6]
        let code = keys[params["key"] as? String ?? ""]
        if action == "key" && code == nil { throw HostError(message: "Unsupported key") }
        let modifierFlags: [String: CGEventFlags] = ["command": .maskCommand, "shift": .maskShift, "option": .maskAlternate, "control": .maskControl]
        let modifiers = params["modifiers"] as? [String] ?? []
        guard modifiers.allSatisfy({ modifierFlags[$0] != nil }) else { throw HostError(message: "Unsupported modifier") }
        var flags: CGEventFlags = []
        for modifier in modifiers { flags.formUnion(modifierFlags[modifier]!) }
        let dx = params["deltaX"] as? Int32 ?? 0
        let dy = params["deltaY"] as? Int32 ?? 0
        guard abs(Int64(dx)) <= 800, abs(Int64(dy)) <= 800 else { throw HostError(message: "Scroll exceeds the limit") }
        try check(view.app)
        let dispatchWindow = try focusedWindow(view.pid)
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == view.pid, CFEqual(dispatchWindow, view.root),
              let dispatchBounds = rect(dispatchWindow), WindowGeometry.matches(dispatchBounds, view.bounds) else {
            throw HostError(message: "The user changed the active window before dispatch.")
        }
        if let start { try checkHit(start, pid: view.pid) }
        if let end { try checkHit(end, pid: view.pid) }
        let scrollPoint = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
        if action == "scroll" { try checkHit(scrollPoint, pid: view.pid) }
        try control.check(epoch)
        uncertain = true
        do {
            switch action {
            case "press":
                guard AXUIElementPerformAction(element!.handle, kAXPressAction as CFString) == .success else {
                    throw HostError(message: "The app did not acknowledge the press.", outcome: "unknown")
                }
            case "set_value":
                guard AXUIElementSetAttributeValue(element!.handle, kAXValueAttribute as CFString, text as CFString) == .success else {
                    throw HostError(message: "The app did not acknowledge the value change.", outcome: "unknown")
                }
            case "click":
                try mouse(.leftMouseDown, at: start!)
                try mouse(.leftMouseUp, at: start!)
            case "drag":
                try mouse(.leftMouseDown, at: start!)
                try mouse(.leftMouseDragged, at: end!)
                try mouse(.leftMouseUp, at: end!)
            case "type": try key(0, text: text)
            case "key": try key(code!, flags: flags)
            case "scroll":
                guard let event = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 2, wheel1: dy, wheel2: dx, wheel3: 0) else {
                    throw HostError(message: "Could not construct scroll input.", outcome: "unknown")
                }
                event.location = scrollPoint
                event.post(tap: .cghidEventTap)
            default: throw HostError(message: "Unsupported action")
            }
            try await Task.sleep(for: .milliseconds(200))
            var result = try await observe(view.app)
            result["actionStatus"] = "dispatched"
            uncertain = false
            return result
        } catch {
            throw HostError(message: "The action may have taken effect. Inspect the app before continuing. \(error)", outcome: "unknown")
        }
    }

    func savedDocument(_ root: AXUIElement) throws -> URL {
        guard let raw = attribute(root, kAXDocumentAttribute) as? String,
              let url = URL(string: raw), url.isFileURL,
              url.host == nil || url.host == "" || url.host == "localhost" else {
            throw HostError(message: "This window does not identify a saved file. Save it, then observe again.")
        }
        guard (attribute(root, "AXEdited") as? Bool) != true else {
            throw HostError(message: "Save the document's pending changes before publishing it.")
        }
        return url.standardizedFileURL
    }

    func documentData(_ url: URL) throws -> Data {
        let fd = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        guard fd >= 0 else { throw HostError(message: "The saved document cannot be opened safely.") }
        let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size <= 16 * 1024 * 1024 else {
            throw HostError(message: "Publish supports regular files up to 16 MB.")
        }
        let data = try file.read(upToCount: 16 * 1024 * 1024 + 1) ?? Data()
        guard data.count == info.st_size else { throw HostError(message: "The saved file changed while reading it.") }
        return data
    }

    func publish(_ view: Observation) async throws -> [String: Any] {
        let process = try running(view.app)
        guard process.processIdentifier == view.pid else { throw HostError(message: "The app restarted. Observe again.") }
        let root = try focusedWindow(view.pid)
        guard CFEqual(root, view.root), let bounds = rect(root), WindowGeometry.matches(bounds, view.bounds) else {
            throw HostError(message: "The document window changed. Observe again.")
        }
        let url = try savedDocument(root)
        let before = try documentData(url)
        guard review("Share this saved document with Rune?", detail: "App: \(view.app)\nFile: \(url.path)\nSize: \(before.count) bytes\n\nA copy of this exact file will be available to download in this Rune conversation. The copy includes all contents of the file.") else {
            throw HostError(message: "Document sharing was declined.")
        }
        try check(view.app)
        guard ProcessInfo.processInfo.systemUptime - view.captured < 150,
              CFEqual(root, try focusedWindow(view.pid)), try savedDocument(root) == url else {
            throw HostError(message: "The document changed during review. Observe again.")
        }
        let data = try documentData(url)
        guard data == before else { throw HostError(message: "The saved file changed during review. Publish it again.") }
        var result = try await observe(view.app)
        try check(view.app)
        result["artifact"] = ["path": url.path, "data_base64": data.base64EncodedString(),
                              "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
        return result
    }

    func handle(_ method: String, _ params: [String: Any]) async throws -> [String: Any] {
        switch method {
        case "status":
            let apps = applications().map { id, url in
                ["id": id, "name": (Bundle(url: url)?.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String)
                    ?? url.deletingPathExtension().lastPathComponent]
            }.sorted { $0["name"]! < $1["name"]! }
            return ["accessibility": AXIsProcessTrusted(), "screenRecording": CGPreflightScreenCaptureAccess(), "apps": apps]
        case "permissions":
            let pane: String
            if !AXIsProcessTrusted() {
                let prompt = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
                _ = AXIsProcessTrustedWithOptions([prompt: true] as CFDictionary)
                pane = "Privacy_Accessibility"
            } else if !CGPreflightScreenCaptureAccess() {
                _ = CGRequestScreenCaptureAccess()
                pane = "Privacy_ScreenCapture"
            } else {
                return ["requested": false, "settingsOpened": false]
            }
            let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane)")!
            if !NSWorkspace.shared.open(url) {
                guard let settings = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.systempreferences"),
                      NSWorkspace.shared.open(settings) else {
                    throw HostError(message: "Could not open System Settings. Open Privacy & Security and enable the required Rune Computer permission.")
                }
            }
            return ["requested": true, "settingsOpened": true, "pane": pane]
        case "grant":
            let apps = params["apps"] as? [String] ?? []
            let installed = applications()
            guard !apps.isEmpty, apps.count <= 12, apps.allSatisfy({ installed[$0] != nil }) else { throw HostError(message: "Invalid app grant") }
            guard review("Allow Rune to access these apps for 30 minutes?",
                         detail: apps.joined(separator: "\n") + "\n\nSelected app windows and text may be sent to your conversation's model provider. Opening an app may bring its window forward. Every input still requires a separate native approval.") else {
                throw HostError(message: "Native app access was declined.")
            }
            try control.check(epoch)
            allowed = Set(apps)
            expires = ProcessInfo.processInfo.systemUptime + 1800
            observation = nil
            uncertain = false
            return ["granted": true]
        case "open":
            let app = params["app"] as? String ?? ""
            try check(app)
            guard let url = applications()[app] else { throw HostError(message: "The installed app is no longer available.") }
            let config = NSWorkspace.OpenConfiguration()
            config.activates = true
            _ = try await NSWorkspace.shared.openApplication(at: url, configuration: config)
            try await Task.sleep(for: .milliseconds(250))
            return try await observe(app)
        case "observe": return try await observe(params["app"] as? String ?? "")
        case "act": return try await act(params)
        case "acknowledge": uncertain = false; observation = nil; return ["acknowledged": true]
        default: throw HostError(message: "Unknown native method")
        }
    }
}
