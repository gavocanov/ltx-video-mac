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
        // Targeted kill by PID only; the runner forwards to its child.
        process.terminate()
        processes.removeAll { $0 === process }
        return true
    }
}
