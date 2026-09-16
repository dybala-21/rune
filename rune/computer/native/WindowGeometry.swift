import CoreGraphics

enum WindowGeometry {
    static func matches(_ lhs: CGRect, _ rhs: CGRect) -> Bool {
        guard !lhs.isEmpty, !rhs.isEmpty, !lhs.isInfinite, !rhs.isInfinite,
              !lhs.isNull, !rhs.isNull else { return false }
        // Accessibility and ScreenCaptureKit can round fractional points differently.
        return abs(lhs.minX - rhs.minX) <= 1 && abs(lhs.minY - rhs.minY) <= 1
            && abs(lhs.width - rhs.width) <= 1 && abs(lhs.height - rhs.height) <= 1
    }
}
