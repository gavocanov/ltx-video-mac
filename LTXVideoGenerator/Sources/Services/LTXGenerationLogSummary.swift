import Foundation

enum LTXGenerationLogSummary {
    static let defaultLogPath = "/tmp/ltx_generation.log"

    static func appendToLog(path: String = defaultLogPath, lines: [String]) {
        let text = lines.joined(separator: "\n") + "\n"
        guard let data = text.data(using: .utf8) else { return }
        if FileManager.default.fileExists(atPath: path) {
            guard let handle = FileHandle(forWritingAtPath: path) else { return }
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: URL(fileURLWithPath: path))
        }
    }

    /// Returns the full generation log for display in the error dialog.
    static func userFacingExcerpt(path: String = defaultLogPath) -> String {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              !data.isEmpty
        else {
            return "(Log missing or empty at \(path).)"
        }
        return String(data: data, encoding: .utf8) ?? ""
    }
}
