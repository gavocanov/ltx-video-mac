import Foundation

/// Tracks all spawned Python processes so they can be terminated on app quit.
final class ProcessRegistry {
    static let shared = ProcessRegistry()

    private var processes: [Process] = []
    private let lock = NSLock()

    private init() {}

    func register(_ process: Process) {
        lock.lock()
        processes.append(process)
        lock.unlock()
    }

    func unregister(_ process: Process) {
        lock.lock()
        processes.removeAll { $0 === process }
        lock.unlock()
    }

    /// Terminate all running processes (called on app termination).
    func terminateAll() {
        lock.lock()
        for process in processes where process.isRunning {
            // Kill by positive PID only (never -pid) so we never touch the
            // app's own process group. The runner forwards the signal to its
            // child (generate_av.py).
            process.terminate()
        }
        processes.removeAll()
        lock.unlock()
    }

    /// Terminate the most recently registered running process (the current
    /// generation). Returns true if a process was killed.
    @discardableResult
    func terminateCurrent() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let process = processes.last(where: { $0.isRunning }) else {
            return false
        }
        // Kill by positive PID only — never -pid (group kill), which would hit
        // the app's shared Python environment. The runner forwards SIGKILL to
        // its own child, so this stops exactly the spawned job.
        process.terminate()
        processes.removeAll { $0 === process }
        return true
    }

    /// Kill the current generation's entire process tree by walking PIDs with
    /// `pgrep -P` (never the process group, which would hit the shared Python
    /// env). Called directly from cancel, independent of task cancellation.
    func killCurrentTree() {
        lock.lock()
        guard let process = processes.last(where: { $0.isRunning }) else {
            lock.unlock()
            return
        }
        let runnerPID = process.processIdentifier
        processes.removeAll { $0 === process }
        lock.unlock()

        var pids = [runnerPID]
        var queue = [runnerPID]
        while !queue.isEmpty {
            let parent = queue.removeFirst()
            for c in Self.childPIDs(of: parent) where !pids.contains(c) {
                pids.append(c)
                queue.append(c)
            }
        }
        for pid in pids.reversed() where pid > 0 {
            kill(pid, SIGKILL)
        }
        process.terminate()
    }

    /// Direct child PIDs of `parent` via `pgrep -P`.
    private static func childPIDs(of parent: Int32) -> [Int32] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        p.arguments = ["-P", "\(parent)"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
        } catch {
            return []
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let str = String(data: data, encoding: .utf8) else { return [] }
        return str.split(whereSeparator: \.isNewline).compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
    }
}
