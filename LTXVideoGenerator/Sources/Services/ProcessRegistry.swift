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
            let pid = process.processIdentifier
            if pid > 0 {
                // Kill the entire process group so child processes die too.
                kill(-pid, SIGKILL)
            }
            process.terminate()
        }
        processes.removeAll()
        lock.unlock()
    }
}
