import Foundation

/// Fake bridge that blocks on generate() until released, so we can cancel
/// mid-generation — exactly the scenario that crashed the app.
final class BlockingBridge: GenerationBridging {
    /// Each generate call blocks until its index is released.
    var releaseFlags: [Bool] = []
    var generateCalls = 0
    var isModelLoaded: Bool { true }
    func loadModel(progressHandler: @escaping (String) -> Void) async throws {}
    func unloadModel() async {}
    func generate(
        request: GenerationRequest,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void,
        previewHandler: ((String) -> Void)?
    ) async throws -> (videoPath: String, seed: Int, enhancedPrompt: String?) {
        let idx = generateCalls
        generateCalls += 1
        if releaseFlags.count <= idx { releaseFlags.append(false) }
        // Block until this specific call is released.
        while !releaseFlags[idx] {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        return (outputPath, 42, nil)
    }
}

@MainActor
func waitUntil(timeout: TimeInterval, _ condition: () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if condition() { return true }
        try? await Task.sleep(nanoseconds: 20_000_000)
    }
    return condition()
}

@MainActor
func runTest() async -> Bool {
    UserDefaults.standard.set("/fake/python", forKey: "pythonPath")
    defer { UserDefaults.standard.removeObject(forKey: "pythonPath") }

    let bridge = BlockingBridge()
    let service = GenerationService(historyManager: HistoryManager(), bridge: bridge)
    // Bypass real Python env check.
    service.pythonEnvCheck = { _ in (true, "ok", nil) }

    let req1 = GenerationRequest(prompt: "first")
    let req2 = GenerationRequest(prompt: "second")
    service.addToQueue(req1)
    service.addToQueue(req2)

    // Wait until req1 is processing (bridge.generate blocked).
    guard await waitUntil(timeout: 5, { service.currentRequest?.id == req1.id }) else {
        print("FAIL: req1 never started processing")
        return false
    }
    print("OK: req1 started processing")

    // Cancel the running job.
    service.cancelCurrent()
    print("OK: cancelCurrent() returned without crashing")

    // The cancelled job must be marked .cancelled (NOT .failed), and no error
    // dialog state may be set.
    let req1Status = service.queue.first { $0.id == req1.id }?.status
    print("DIAG: req1 status after cancel = \(String(describing: req1Status)); error = \(String(describing: service.error))")
    if req1Status != .cancelled {
        print("FAIL: cancelled job not marked .cancelled (got \(String(describing: req1Status)))")
        return false
    }
    if service.error != nil {
        print("FAIL: error dialog state set on cancel: \(String(describing: service.error))")
        return false
    }
    print("OK: cancelled job marked .cancelled, no error dialog")

    // Release req1's blocked generate so the cancelled task can unwind.
    bridge.releaseFlags[0] = true

    // The next queued job (req2) must start automatically — verify by waiting
    // for a SECOND generate call, while req2 stays blocked.
    guard await waitUntil(timeout: 5, { bridge.generateCalls >= 2 }) else {
        print("FAIL: next job did not start after cancel (queue stalled); generateCalls=\(bridge.generateCalls)")
        return false
    }
    print("OK: next job (req2) started after cancel (generateCalls=2)")

    // Release req2 so it completes and the queue drains.
    bridge.releaseFlags[1] = true

    // Both jobs should eventually be removed from the queue.
    guard await waitUntil(timeout: 5, { service.queue.isEmpty }) else {
        print("FAIL: queue did not drain; remaining=\(service.queue.map { ($0.prompt, $0.status) })")
        return false
    }
    print("OK: queue drained")

    guard bridge.generateCalls == 2 else {
        print("FAIL: expected 2 generate calls, got \(bridge.generateCalls)")
        return false
    }
    print("OK: both jobs processed (generateCalls=2)")

    print("\nPASS")
    return true
}

// Entry point
@main
struct Main {
    static func main() async {
        let ok = await runTest()
        exit(ok ? 0 : 1)
    }
}
