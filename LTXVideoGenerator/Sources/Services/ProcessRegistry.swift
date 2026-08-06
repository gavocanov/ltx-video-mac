import Foundation

/// Tracks all spawned Python processes so they can be terminated on app quit.
final class ProcessRegistry {
    static let shared = ProcessRegistry()

    private var processes: [Process] = []
    /// Child PIDs announced by each runner (CHILD_PID=...), keyed by the
    /// runner's Process identity. These are authoritative — pgrep is unreliable.
    private var childPIDsByProcess: [ObjectIdentifier: Int32] = [:]
    /// Runner PIDs that were killed by cancel. runPython consults this to
    /// distinguish "killed by user cancel" from a real failure, so no spurious
    /// "generation failed" dialog appears. Set synchronously in killCurrentTree,
    /// independent of async task-cancellation callbacks.
    private var cancelledRunnerPIDs: Set<Int32> = []
    private let lock = NSLock()

    private init() {}

    /// True if the given runner PID was killed by a user cancel.
    func wasCancelled(runnerPID: Int32) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelledRunnerPIDs.contains(runnerPID)
    }

    func register(_ process: Process) {
        lock.lock()
        processes.append(process)
        lock.unlock()
    }

    func unregister(_ process: Process) {
        lock.lock()
        processes.removeAll { $0 === process }
        childPIDsByProcess[ObjectIdentifier(process)] = nil
        lock.unlock()
    }

    /// Record the generation subprocess PID announced by a runner. This is the
    /// authoritative child PID used for cancellation (pgrep is unreliable).
    func registerChildPID(_ pid: Int32, for process: Process) {
        lock.lock()
        childPIDsByProcess[ObjectIdentifier(process)] = pid
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

    /// Kill the current generation's process tree by walking PIDs with
    /// `pgrep -P` (never the process group, which would hit the shared Python
    /// env). Called directly from cancel, independent of task cancellation.
    func killCurrentTree() {
        lock.lock()
        guard let process = processes.last(where: { $0.isRunning }) else {
            lock.unlock()
            return
        }
        let runnerPID = process.processIdentifier
        let trackedChildPID = childPIDsByProcess[ObjectIdentifier(process)]
        processes.removeAll { $0 === process }
        childPIDsByProcess[ObjectIdentifier(process)] = nil
        // Mark this runner as cancelled BEFORE killing, so runPython can tell a
        // user cancel apart from a real failure (no spurious error dialog).
        cancelledRunnerPIDs.insert(runnerPID)
        lock.unlock()

        // Only kill the runner and its generation subprocess — never walk
        // further up or use a group kill, so the app's shared Python env is
        // untouched. The tracked CHILD_PID is authoritative (announced by the
        // runner); pgrep proved unreliable, so we do not fall back to it.
        var pids = [runnerPID]
        if let trackedChildPID, trackedChildPID > 0, !pids.contains(trackedChildPID) {
            pids.append(trackedChildPID)
        }
        for pid in pids.reversed() where pid > 0 {
            kill(pid, SIGKILL)
        }
        process.terminate()
    }
}
