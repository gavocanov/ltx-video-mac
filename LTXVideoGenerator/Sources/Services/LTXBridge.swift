import Foundation

enum LTXError: LocalizedError, Equatable {
    case pythonNotConfigured
    case modelLoadFailed(String)
    case generationFailed(String)
    case exportFailed(String)
    case cancelled
    
    var errorDescription: String? {
        switch self {
        case .pythonNotConfigured:
            return "Python environment not configured. Please check Preferences."
        case .modelLoadFailed(let msg):
            return "Failed to load LTX model: \(msg)"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        case .exportFailed(let msg):
            return "Failed to export video: \(msg)"
        case .cancelled:
            return "Generation was cancelled"
        }
    }
}

// Use subprocess to run MLX-based generation
class LTXBridge {
    static let shared = LTXBridge()

    private static func stderrIndicatesMetalInteractivity(_ stderr: String) -> Bool {
        let low = stderr.lowercased()
        return low.contains("diagnostic_metal_interactivity")
            || low.contains("impacting interactivity")
            || low.contains("kiogpucommandbuffercallbackerrorimpactinginteractivity")
    }

    private(set) var isModelLoaded = false
    private var pythonHome: String?
    private var pythonExecutable: String?
    
    private init() {
        setupPythonPaths()
    }
    
    private func setupPythonPaths() {
        // Get Python path from user defaults
        guard let savedPath = UserDefaults.standard.string(forKey: "pythonPath"),
              !savedPath.isEmpty else {
            pythonExecutable = nil
            pythonHome = nil
            return
        }
        
        // Use PythonEnvironment's path detection to handle both executable and dylib paths
        let pathType = PythonEnvironment.shared.detectPathType(savedPath)
        
        switch pathType {
        case .executable:
            pythonExecutable = savedPath
            if let dylib = PythonEnvironment.shared.executableToDylib(savedPath),
               let home = PythonEnvironment.shared.extractPythonHome(from: dylib) {
                pythonHome = home
            } else {
                let execURL = URL(fileURLWithPath: savedPath)
                pythonHome = execURL.deletingLastPathComponent().deletingLastPathComponent().path
            }
            
        case .dylib:
            if let exec = PythonEnvironment.shared.dylibToExecutable(savedPath) {
                pythonExecutable = exec
            }
            if let home = PythonEnvironment.shared.extractPythonHome(from: savedPath) {
                pythonHome = home
                if pythonExecutable == nil {
                    let standardExec = "\(home)/bin/python3"
                    if FileManager.default.isExecutableFile(atPath: standardExec) {
                        pythonExecutable = standardExec
                    }
                }
            }
            
        case .unknown:
            if FileManager.default.isExecutableFile(atPath: savedPath) {
                pythonExecutable = savedPath
                let execURL = URL(fileURLWithPath: savedPath)
                pythonHome = execURL.deletingLastPathComponent().deletingLastPathComponent().path
            } else {
                pythonExecutable = nil
                pythonHome = nil
            }
        }
    }
    
    func loadModel(progressHandler: @escaping (String) -> Void) async throws {
        setupPythonPaths()
        
        guard pythonExecutable != nil else {
            throw LTXError.pythonNotConfigured
        }
        
        progressHandler("Checking MLX environment...")
        
        // Test that MLX and required packages are installed
        let testScript = """
        import mlx.core as mx
        import mlx_vlm
        import transformers
        print("OK")
        """
        
        let result = try await runPython(script: testScript)
        if !result.contains("OK") {
            throw LTXError.pythonNotConfigured
        }
        
        let selectedModel = LTXModelCatalog.selectedModel()
        progressHandler("MLX environment ready. Model will download on first generation (\(selectedModel.downloadSize)).")
        isModelLoaded = true
    }
    
    func generate(
        request: GenerationRequest,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void,
        previewHandler: ((String) -> Void)? = nil
    ) async throws -> (videoPath: String, seed: Int, enhancedPrompt: String?) {
        setupPythonPaths()
        
        guard let _ = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }
        
        let params = request.parameters
        let seed = params.seed ?? Int.random(in: 0..<Int(Int32.max))
        
        let selectedModel = LTXModelCatalog.resolvedModel(id: request.modelId)
        let modelRepo = selectedModel.repo
        let selectedTextEncoder = LTXTextEncoderCatalog.resolvedTextEncoder(id: request.textEncoderId)
        let textEncoderRepo = selectedTextEncoder.repo
        guard !textEncoderRepo.isEmpty else {
            throw LTXError.generationFailed(
                "Set a text encoder Hugging Face repo in Preferences → General (pick a preset or fill in Custom)."
            )
        }
        let (effectiveTilingMode, appliedTilingRecovery) = GenerationFailureRecovery.effectiveTilingMode(
            requested: params.vaeTilingMode
        )
        if appliedTilingRecovery {
            progressHandler(0.05, "VAE tiling set to Auto after a previous Metal timeout with Aggressive tiling.")
        }
        let isImageToVideo = request.isImageToVideo
        let modeDescription = isImageToVideo ? "image-to-video" : "text-to-video"
        progressHandler(0.1, "Starting \(modeDescription) (\(selectedModel.displayName))...")
        if selectedModel.supportsBuiltInAudio && !request.disableAudio && params.fps != 24 {
            progressHandler(0.1, "Sync tip: speech alignment works best at 24 FPS (current: \(params.fps))")
        }
        
        let enableGemmaPromptEnhancement = UserDefaults.standard.bool(forKey: "enableGemmaPromptEnhancement")
        let saveAudioTrackSeparately = UserDefaults.standard.bool(forKey: "saveAudioTrackSeparately")
        let useLocalMlxVideoRepoPref = UserDefaults.standard.bool(forKey: "useLocalMlxVideoRepo")
        // Default cadence is 3; only honor an explicit stored value.
        let previewEvery: Int
        if UserDefaults.standard.object(forKey: "previewEvery") != nil {
            previewEvery = max(0, UserDefaults.standard.integer(forKey: "previewEvery"))
        } else {
            previewEvery = 3
        }
        // Per-generation temp dir for latent preview frames.
        let previewDir = NSTemporaryDirectory() + "ltx-preview-" + UUID().uuidString
        try? FileManager.default.createDirectory(
            atPath: previewDir,
            withIntermediateDirectories: true
        )
        // Always remove the preview dir when generation finishes (success, error, or cancel).
        defer {
            try? FileManager.default.removeItem(atPath: previewDir)
        }

        // Apply prompt enhancement up-front so generation can continue safely even
        // when upstream enhancer internals fail.
        var preEnhancedPrompt: String? = nil
        var generationPrompt = request.prompt
        if enableGemmaPromptEnhancement {
            progressHandler(0.06, "Enhancing prompt...")
            do {
                if let enhanced = try await previewEnhancedPrompt(
                    prompt: request.prompt,
                    modelRepo: modelRepo,
                    temperature: request.gemmaTopP,
                    sourceImagePath: request.sourceImagePath,
                    progressHandler: { status in
                    progressHandler(0.06, status)
                    }
                ), !enhanced.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    generationPrompt = enhanced
                    preEnhancedPrompt = enhanced
                    progressHandler(0.07, "Prompt enhanced with Gemma")
                } else {
                    progressHandler(0.07, "Prompt enhancement returned empty text; using original prompt")
                }
            } catch {
                progressHandler(0.07, "Prompt enhancement failed; using original prompt")
            }
        }

        // Log file path: same folder and base name as the output video, .log extension.
        let logFile = (outputPath as NSString).deletingPathExtension + ".log"
        
        // Ensure dimensions are divisible by 64 for MLX
        let genWidth = (params.width / 64) * 64
        let genHeight = (params.height / 64) * 64
        
        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"

        // Run our vendored generate_av_runner.py (a standalone script) instead of
        // embedding Python in Swift, so we avoid Swift/Python interpolation bugs.
        let runnerScript = resourcesPath + "/generate_av_runner.py"
        guard FileManager.default.fileExists(atPath: runnerScript) else {
            throw LTXError.generationFailed("Generation runner script not found at \(runnerScript)")
        }
        var scriptArgs = [
            "--log-file", logFile,
            "--model-repo", modelRepo,
            "--text-encoder-repo", textEncoderRepo,
            "--use-local-pref", useLocalMlxVideoRepoPref ? "1" : "0",
            "--image-path", request.sourceImagePath ?? "",
            "--prompt", generationPrompt,
            "--negative-prompt", request.negativePrompt,
            "--width", String(genWidth),
            "--height", String(genHeight),
            "--num-frames", String(params.numFrames),
            "--seed", String(seed),
            "--fps", String(params.fps),
            "--steps", String(params.numInferenceSteps),
            "--cfg-scale", String(params.guidanceScale),
            "--output-path", outputPath,
            "--tiling", effectiveTilingMode,
            "--preview-every", String(previewEvery),
            "--preview-dir", previewDir,
            "--resources-path", resourcesPath,
        ]
        if request.disableAudio { scriptArgs.append("--disable-audio") }
        if saveAudioTrackSeparately { scriptArgs.append("--save-audio-separately") }
        if let img = request.sourceImagePath, !img.isEmpty {
            scriptArgs.append(contentsOf: ["--image-strength", String(params.imageStrength)])
        }
        if let lora = params.loraPath, !lora.isEmpty {
            scriptArgs.append(contentsOf: ["--lora-path", lora])
            scriptArgs.append(contentsOf: ["--lora-strength", String(params.loraStrength)])
        }

        progressHandler(0.05, "Running MLX generation...")

        // Thread-safe capture of enhanced prompt from stderr
        let enhancedPromptLock = NSLock()
        var capturedEnhancedPrompt: String? = preEnhancedPrompt
        let stderrLineBufferLock = NSLock()
        var stderrLineBuffer = ""

        let output: String
        do {
            output = try await runPython(
                script: runnerScript,
                scriptArgs: scriptArgs,
                timeout: 21600, // 6h: model download + generation can exceed 1h on slow links
                generationDiagnostics: (modelRepo: modelRepo, textEncoderRepo: textEncoderRepo),
                originalVaeTilingMode: request.parameters.vaeTilingMode,
                logFile: logFile
            ) { stderrChunk in
            // Build complete logical lines from chunked stderr reads so STAGE/STATUS tokens
            // are never dropped when a token is split across read boundaries.
            stderrLineBufferLock.lock()
            stderrLineBuffer += stderrChunk.replacingOccurrences(of: "\r", with: "\n")
            let completeLines = stderrLineBuffer.components(separatedBy: "\n")
            stderrLineBuffer = completeLines.last ?? ""
            let linesToParse = Array(completeLines.dropLast())
            stderrLineBufferLock.unlock()

            guard !linesToParse.isEmpty else { return }

            // Capture enhanced prompt from stderr
            // Our generate.py emits "ENHANCED_PROMPT:..." and mlx_video may emit "Enhanced prompt: ..."
            for line in linesToParse {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                var extracted: String? = nil
                if trimmed.hasPrefix("ENHANCED_PROMPT:") {
                    extracted = String(trimmed.dropFirst("ENHANCED_PROMPT:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                } else if trimmed.lowercased().hasPrefix("enhanced prompt:") {
                    extracted = String(trimmed.dropFirst("enhanced prompt:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                }
                if let text = extracted, !text.isEmpty {
                    enhancedPromptLock.lock()
                    capturedEnhancedPrompt = text
                    enhancedPromptLock.unlock()
                }
            }
            
            DispatchQueue.main.async {
                // Parse complete lines only; chunk boundary handling is done above.
                for raw in linesToParse {
                    let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    if line.isEmpty { continue }
                    let cleanLine = line.replacingOccurrences(
                        of: #"\u{001B}\[[0-9;]*[A-Za-z]"#,
                        with: "",
                        options: .regularExpression
                    )
                    let lower = cleanLine.lowercased()

                    if cleanLine.hasPrefix("DOWNLOAD:STALL:") {
                        let seconds = String(cleanLine.dropFirst("DOWNLOAD:STALL:".count))
                        progressHandler(0.01, "Download stalled for \(seconds)s. Stopping generation.")
                    } else if lower.contains("diagnostic_metal_interactivity")
                                || lower.contains("kiogpucommandbuffercallbackerrorimpactinginteractivity")
                                || lower.contains("impacting interactivity") {
                        progressHandler(0.01, "Generation stopped: Metal watchdog timeout")
                        if request.parameters.vaeTilingMode == "aggressive" {
                            GenerationFailureRecovery.recordMetalInteractivityFailureWithAggressiveTiling()
                        }
                    } else if lower.contains("kiogpucommandbuffercallbackerroroutofmemory")
                                || lower.contains("insufficient memory")
                                || lower.contains("std::bad_alloc")
                                || (lower.contains("command buffer execution failed") && lower.contains("memory"))
                                || (lower.contains("metal") && lower.contains("out of memory")) {
                        progressHandler(0.01, "Generation stopped: GPU memory limit reached")
                    }

                    if cleanLine.hasPrefix("STAGE:") {
                        // Parse stage-aware progress: STAGE:1:STEP:3:8:Denoising
                        let parts = cleanLine.components(separatedBy: ":")
                        if parts.count >= 5,
                           let stage = Int(parts[1]),
                           let step = Int(parts[3]),
                           let total = Int(parts[4]) {
                            let stageProgress = Double(step) / Double(total)
                            let mappedProgress: Double
                            let message: String
                            
                            if stage == 0 {
                                // Download phase: map file index/total into 0.01...0.08
                                mappedProgress = 0.01 + (stageProgress * 0.07)
                                message = parts.count > 5
                                    ? String(parts[5...].joined(separator: ":"))
                                    : "Downloading model files (\(step)/\(total))"
                            } else if stage == 1 {
                                mappedProgress = 0.1 + (stageProgress * 0.4)
                                message = "Stage 1 (\(step)/\(total)): Generating at half resolution"
                            } else {
                                mappedProgress = 0.5 + (stageProgress * 0.4)
                                message = "Stage 2 (\(step)/\(total)): Refining at full resolution"
                            }
                            progressHandler(mappedProgress, message)
                        }
                    } else if cleanLine.hasPrefix("STATUS:") {
                        let message = String(cleanLine.dropFirst(7))
                        if message.contains("Stage 1") {
                            progressHandler(0.1, message)
                        } else if message.contains("Stage 2") || message.contains("Upsampling") {
                            progressHandler(0.5, message)
                        } else if message.contains("Decoding") {
                            progressHandler(0.9, message)
                        } else if message.contains("Saving") {
                            progressHandler(0.95, message)
                        } else if message.contains("Loading") {
                            progressHandler(0.08, message)
                        } else {
                            progressHandler(0.05, message)
                        }
                    } else if let stageMatch = lower.firstMatch(of: #/stage\s+([12])\s*\((\d+)\/(\d+)\)/#),
                              let stage = Int(stageMatch.1),
                              let step = Int(stageMatch.2),
                              let total = Int(stageMatch.3),
                              total > 0 {
                        let stageProgress = Double(step) / Double(total)
                        let mappedProgress = stage == 1
                            ? 0.1 + (stageProgress * 0.4)
                            : 0.5 + (stageProgress * 0.4)
                        let message = stage == 1
                            ? "Stage 1 (\(step)/\(total)): Generating at half resolution"
                            : "Stage 2 (\(step)/\(total)): Refining at full resolution"
                        progressHandler(mappedProgress, message)
                    } else if cleanLine.hasPrefix("MLX_VIDEO_VERSION:") {
                        let version = String(cleanLine.dropFirst(18))
                        print("[LTXBridge] mlx-video-with-audio v\(version)")
                    } else if cleanLine.hasPrefix("MODEL:CACHED:") {
                        let repo = String(cleanLine.dropFirst(13))
                        progressHandler(0.08, "Model cached: \(repo)")
                    } else if cleanLine.hasPrefix("DOWNLOAD:START:") {
                        let repo = String(cleanLine.dropFirst(15))
                        progressHandler(0.01, "Downloading model: \(repo)")
                    } else if cleanLine.hasPrefix("DOWNLOAD:HEARTBEAT:") {
                        let repo = String(cleanLine.dropFirst(18))
                        progressHandler(0.04, "Downloading \(repo)… (still working — large files can look idle)")
                    } else if cleanLine.hasPrefix("DOWNLOAD:PROGRESS:") {
                        let parts = cleanLine.dropFirst(18).split(separator: ":")
                        if parts.count >= 4 {
                            let currentFile = Int(parts[0]) ?? 0
                            let totalFiles = Int(parts[1]) ?? 1
                            let repo = String(parts[2])
                            let filename = String(parts[3...].joined(separator: ":"))
                            let pct = totalFiles > 0 ? Double(currentFile) / Double(totalFiles) : 0
                            let mappedProgress = 0.01 + (pct * 0.07)
                            progressHandler(mappedProgress, "Downloading \(repo) (\(currentFile)/\(totalFiles)): \(filename)")
                        }
                    } else if cleanLine.hasPrefix("DOWNLOAD:COMPLETE:") {
                        progressHandler(0.08, "Model download complete")
                    } else if cleanLine.hasPrefix("PREVIEW:ENABLED:") {
                        print("[LTXBridge] \(cleanLine)")
                    } else if cleanLine.hasPrefix("PREVIEW:WRITE:") {
                        print("[LTXBridge] \(cleanLine)")
                    } else if cleanLine.hasPrefix("PREVIEW:") {
                        let path = String(cleanLine.dropFirst("PREVIEW:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                        if !path.isEmpty {
                            print("[LTXBridge] preview frame: \(path)")
                            previewHandler?(path)
                        }
                    } else if cleanLine.contains("Downloading") || cleanLine.contains("Fetching") {
                        // huggingface_hub tqdm output
                        let fileCountPattern = #/(\d+)%\|[^|]*\|\s*(\d+)/(\d+)/#
                        if let match = cleanLine.firstMatch(of: fileCountPattern) {
                            let currentFile = Int(match.2) ?? 0
                            let totalFiles = Int(match.3) ?? 1
                            var filePercent = Double(currentFile) / Double(max(totalFiles, 1))
                            var message = "Downloading: \(currentFile)/\(totalFiles) files"
                            if let bytesMatch = cleanLine.firstMatch(of: #/\|\s*([\d.]+)([KMG]?)B?\/([\d.]+)([KMG]?)B?/#) {
                                let curVal = Double(bytesMatch.1) ?? 0
                                let totVal = Double(bytesMatch.3) ?? 1
                                let unit = String(bytesMatch.2)
                                let scale: Double = unit == "G" ? 1 : (unit == "M" ? 0.001 : 0.000001)
                                let curGB = curVal * scale
                                let totGB = totVal * scale
                                let pct = totVal > 0 ? Int(100 * curVal / totVal) : 0
                                filePercent = (Double(currentFile) + Double(pct) / 100.0) / Double(max(totalFiles, 1))
                                message = String(format: "Downloading: %.1fGB / %.1fGB (file %d/%d, %d%%)", curGB, totGB, currentFile + 1, totalFiles, pct)
                            }
                            let mappedProgress = 0.01 + (filePercent * 0.07)
                            progressHandler(mappedProgress, message)
                        }
                    }
                }
            }
            }
        } catch {
            let excerpt = LTXGenerationLogSummary.userFacingExcerpt()
            LTXGenerationLogSummary.appendToLog(
                lines: ["", "=== Swift failure ===", error.localizedDescription, "", excerpt]
            )
            let extra = excerpt.isEmpty ? "" : "\n\n--- Full log ---\n" + excerpt
            throw LTXError.generationFailed(error.localizedDescription + extra)
        }
        
        // Parse JSON output - extract JSON from output (may have other text before it)
        // Look for JSON object starting with { and ending with }
        if let jsonStart = output.range(of: "{\"video_path\""),
           let jsonEnd = output.range(of: "}", range: jsonStart.lowerBound..<output.endIndex) {
            let jsonString = String(output[jsonStart.lowerBound...jsonEnd.lowerBound])
            if let data = jsonString.data(using: String.Encoding.utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let videoPath = json["video_path"] as? String,
               let resultSeed = json["seed"] as? Int {
                progressHandler(1.0, "Complete!")
                // Safe to read without lock: runPython has completed, no more stderr callbacks
                return (videoPath, resultSeed, capturedEnhancedPrompt)
            }
        }
        
        throw LTXError.generationFailed("Failed to parse generation output: \(output)")
    }
    
    func unloadModel() async {
        isModelLoaded = false
    }

    /// Preview enhanced prompt without running generation. Returns enhanced text or nil on error.
    func previewEnhancedPrompt(
        prompt: String,
        modelRepo: String,
        temperature: Double,
        sourceImagePath: String?,
        progressHandler: @escaping (String) -> Void
    ) async throws -> String? {
        setupPythonPaths()
        guard let python = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }
        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"
        let scriptPath = resourcesPath + "/enhance_prompt_preview.py"
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            throw LTXError.generationFailed("Preview script not found")
        }
        var args = [
            scriptPath,
            "--prompt", prompt,
            "--model-repo", modelRepo,
            "--temperature", String(temperature),
            "--resources-path", resourcesPath,
        ]
        if let img = sourceImagePath, !img.isEmpty {
            args.append(contentsOf: ["--image", img])
        }
        progressHandler("Loading prompt enhancer (first run may download ~7GB)...")
        let output = try await runPythonScript(
            executable: python,
            arguments: args,
            timeout: 21600, // 6h: first-run model download can be large
            stderrHandler: { chunk in
                // Parse DOWNLOAD:/STATUS: tokens into user-facing progress.
                for line in chunk.split(separator: "\n") {
                    let clean = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
                    if clean.hasPrefix("DOWNLOAD:START:") {
                        let repo = String(clean.dropFirst("DOWNLOAD:START:".count))
                        DispatchQueue.main.async {
                            progressHandler("Downloading prompt enhancer model (\(repo))...")
                        }
                    } else if clean.hasPrefix("DOWNLOAD:PROGRESS:") {
                        let parts = clean.dropFirst("DOWNLOAD:PROGRESS:".count).split(separator: ":")
                        if parts.count >= 4 {
                            let currentFile = Int(parts[0]) ?? 0
                            let totalFiles = Int(parts[1]) ?? 1
                            let filename = String(parts[3...].joined(separator: ":"))
                            DispatchQueue.main.async {
                                progressHandler("Downloading prompt enhancer model (\(currentFile)/\(totalFiles)): \(filename)")
                            }
                        }
                    } else if clean.hasPrefix("DOWNLOAD:BYTES:") {
                        let parts = clean.dropFirst("DOWNLOAD:BYTES:".count).split(separator: ":")
                        if parts.count >= 3,
                           let pct = Int(parts[0]),
                           let done = Double(parts[1]),
                           let totalBytes = Double(parts[2]),
                           totalBytes > 0 {
                            let mbDone = done / 1_048_576
                            let mbTotal = totalBytes / 1_048_576
                            DispatchQueue.main.async {
                                progressHandler(String(format: "Downloading prompt enhancer model… %.1f%% (%.0f / %.0f MB)", Double(pct), mbDone, mbTotal))
                            }
                        }
                    } else if clean.hasPrefix("DOWNLOAD:COMPLETE:") {
                        DispatchQueue.main.async {
                            progressHandler("Prompt enhancer model downloaded. Loading...")
                        }
                    } else if clean.hasPrefix("CLEANED:") {
                        let filename = String(clean.dropFirst("CLEANED:".count))
                        DispatchQueue.main.async {
                            progressHandler("Cleaning interrupted download: \(filename)")
                        }
                    } else if clean.hasPrefix("FILE_FAILED:") {
                        let rest = String(clean.dropFirst("FILE_FAILED:".count))
                        DispatchQueue.main.async {
                            progressHandler("Download issue: \(rest)")
                        }
                    } else if clean.hasPrefix("PREDOWNLOAD_ERROR:") {
                        let rest = String(clean.dropFirst("PREDOWNLOAD_ERROR:".count))
                        DispatchQueue.main.async {
                            progressHandler("Download failed: \(rest)")
                        }
                    } else if clean.hasPrefix("STATUS:") {
                        let message = String(clean.dropFirst("STATUS:".count))
                        DispatchQueue.main.async {
                            progressHandler(message)
                        }
                    }
                }
            }
        )
        if let data = output.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let enhanced = json["enhanced_prompt"] as? String, !enhanced.isEmpty {
                return enhanced
            }
            if let err = json["error"] as? String {
                throw LTXError.generationFailed(err)
            }
        }
        return nil
    }

    private func runPythonScript(
        executable: String,
        arguments: [String],
        timeout: TimeInterval = 60,
        stderrHandler: ((String) -> Void)? = nil
    ) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: executable)
                process.arguments = arguments
                var env: [String: String] = [:]
                let pythonBin = URL(fileURLWithPath: executable).deletingLastPathComponent().path
                env["PATH"] = "\(pythonBin):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
                env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                env["MTL_DEVICE_WRAPPER_TYPE"] = "1"
                // Forward SSL/CA and Hugging Face settings so downloads work like in a terminal.
                let forwardKeys = [
                    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                    "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME", "HF_HUB_CACHE",
                    "HF_HUB_DISABLE_TLS_VERIFICATION", "UV_SYSTEM_CERTS"
                ]
                for key in forwardKeys {
                    if let value = ProcessInfo.processInfo.environment[key], !value.isEmpty {
                        env[key] = value
                    }
                }
                // Apply Hugging Face settings from Preferences (override inherited values).
                let defaults = UserDefaults.standard
                if let token = defaults.string(forKey: "hfToken"), !token.isEmpty {
                    env["HF_TOKEN"] = token
                }
                if let caPath = defaults.string(forKey: "caBundlePath"), !caPath.isEmpty {
                    env["SSL_CERT_FILE"] = caPath
                    env["REQUESTS_CA_BUNDLE"] = caPath
                }
                if defaults.bool(forKey: "disableTLSVerification") {
                    env["HF_HUB_DISABLE_TLS_VERIFICATION"] = "1"
                }
                process.environment = env
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe
                if let stderrHandler {
                    stderrPipe.fileHandleForReading.readabilityHandler = { handle in
                        let data = handle.availableData
                        if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                            stderrHandler(str)
                        }
                    }
                }
                do {
                    try process.run()
                    // Put in its own process group so we can kill the whole tree on quit.
                    setpgid(process.processIdentifier, process.processIdentifier)
                    ProcessRegistry.shared.register(process)
                    // Enforce the timeout: kill the process tree if it exceeds the limit.
                    let timedOut = DispatchSemaphore(value: 0)
                    let timer = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
                    timer.schedule(deadline: .now() + timeout)
                    timer.setEventHandler {
                        process.terminate()
                        // Kill the whole process group (children may outlive the parent).
                        kill(-process.processIdentifier, SIGKILL)
                        timedOut.signal()
                    }
                    timer.resume()
                    process.waitUntilExit()
                    timer.cancel()
                    ProcessRegistry.shared.unregister(process)
                    let didTimeout = timedOut.wait(timeout: .now()) == .success
                    if didTimeout {
                        continuation.resume(throwing: LTXError.generationFailed("Timed out after \(Int(timeout))s"))
                        return
                    }
                    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: outputData, encoding: .utf8) ?? ""
                    let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    if process.terminationStatus != 0 {
                        let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                        let errStr = String(data: errData, encoding: .utf8) ?? ""
                        continuation.resume(throwing: LTXError.generationFailed(errStr.isEmpty ? "Exit code \(process.terminationStatus)" : errStr))
                    } else {
                        continuation.resume(returning: trimmed)
                    }
                } catch {
                    continuation.resume(throwing: LTXError.generationFailed(error.localizedDescription))
                }
            }
        }
    }

    private func runPython(
        script: String,
        scriptArgs: [String] = [],
        timeout: TimeInterval = 60,
        generationDiagnostics: (modelRepo: String, textEncoderRepo: String)? = nil,
        originalVaeTilingMode: String? = nil,
        logFile: String = "/tmp/ltx_generation.log",
        stderrHandler: ((String) -> Void)? = nil
    ) async throws -> String {
        guard let python = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }

        let logFile = logFile

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: python)
                if scriptArgs.isEmpty {
                    process.arguments = ["-c", script]
                } else {
                    process.arguments = [script] + scriptArgs
                }
                
                // Clean environment for MLX
                var env: [String: String] = [:]
                
                let pythonBin = URL(fileURLWithPath: python).deletingLastPathComponent().path
                env["PATH"] = "\(pythonBin):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
                env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                
                // MLX uses Metal - inherit any Metal-related env vars
                if let metalDevice = ProcessInfo.processInfo.environment["MTL_DEVICE_WRAPPER_TYPE"] {
                    env["MTL_DEVICE_WRAPPER_TYPE"] = metalDevice
                }

                // Forward SSL/CA and Hugging Face settings so downloads work like in a terminal.
                let forwardKeys = [
                    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                    "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME", "HF_HUB_CACHE",
                    "HF_HUB_DISABLE_TLS_VERIFICATION", "UV_SYSTEM_CERTS"
                ]
                for key in forwardKeys {
                    if let value = ProcessInfo.processInfo.environment[key], !value.isEmpty {
                        env[key] = value
                    }
                }
                // Apply Hugging Face settings from Preferences (override inherited values).
                let defaults = UserDefaults.standard
                if let token = defaults.string(forKey: "hfToken"), !token.isEmpty {
                    env["HF_TOKEN"] = token
                }
                if let caPath = defaults.string(forKey: "caBundlePath"), !caPath.isEmpty {
                    env["SSL_CERT_FILE"] = caPath
                    env["REQUESTS_CA_BUNDLE"] = caPath
                }
                if defaults.bool(forKey: "disableTLSVerification") {
                    env["HF_HUB_DISABLE_TLS_VERIFICATION"] = "1"
                }

                process.environment = env
                
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe
                
                var stderrAccumulated = ""
                let stderrLock = NSLock()
                
                stderrPipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                        stderrLock.lock()
                        stderrAccumulated += str
                        stderrLock.unlock()
                        
                        if let logData = ("[STDERR] " + str).data(using: .utf8) {
                            if FileManager.default.fileExists(atPath: logFile) {
                                if let handle = FileHandle(forWritingAtPath: logFile) {
                                    handle.seekToEndOfFile()
                                    handle.write(logData)
                                    handle.closeFile()
                                }
                            } else {
                                try? logData.write(to: URL(fileURLWithPath: logFile))
                            }
                        }
                        
                        // Send only the latest chunk; caller parses line-by-line.
                        stderrHandler?(str)
                    }
                }
                
                do {
                    let startLog = "=== LTX MLX Process Started ===\nPython: \(python)\nTime: \(Date())\n"
                    try? startLog.write(toFile: logFile, atomically: false, encoding: .utf8)
                    
                    try process.run()
                    // Put in its own process group so we can kill the whole tree on quit.
                    setpgid(process.processIdentifier, process.processIdentifier)
                    ProcessRegistry.shared.register(process)
                    process.waitUntilExit()
                    ProcessRegistry.shared.unregister(process)

                    stderrPipe.fileHandleForReading.readabilityHandler = nil
                    
                    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: outputData, encoding: .utf8) ?? ""
                    
                    let outputLog = "\n[STDOUT] \(output)\n[EXIT CODE] \(process.terminationStatus)\n"
                    if let handle = FileHandle(forWritingAtPath: logFile) {
                        handle.seekToEndOfFile()
                        handle.write(outputLog.data(using: .utf8)!)
                        handle.closeFile()
                    }
                    
                    let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let data = trimmedOutput.data(using: .utf8),
                       let _ = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        continuation.resume(returning: trimmedOutput)
                        return
                    }
                    
                    if process.terminationStatus != 0 {
                        stderrLock.lock()
                        let stderr = stderrAccumulated
                        stderrLock.unlock()
                        
                        let harmlessPatterns = ["UserWarning", "FutureWarning"]
                        let isOnlyHarmless = harmlessPatterns.allSatisfy { stderr.contains($0) } ||
                                            stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        
                        if !trimmedOutput.isEmpty && isOnlyHarmless {
                            continuation.resume(returning: trimmedOutput)
                        } else {
                            let excerpt = LTXGenerationLogSummary.userFacingExcerpt()
                            if Self.stderrIndicatesMetalInteractivity(stderr),
                               originalVaeTilingMode == "aggressive" {
                                GenerationFailureRecovery.recordMetalInteractivityFailureWithAggressiveTiling()
                            }
                            var message = "Exit code \(process.terminationStatus). Check /tmp/ltx_generation.log"
                            if !stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                message += "\n\n--- Recent stderr ---\n"
                                    + String(stderr.suffix(8000))
                            }
                            if !excerpt.isEmpty {
                                message += "\n\n--- Full log ---\n" + excerpt
                            }
                            LTXGenerationLogSummary.appendToLog(
                                lines: ["", "=== Swift runner summary ===", message]
                            )
                            continuation.resume(throwing: LTXError.generationFailed(message))
                        }
                    } else {
                        continuation.resume(returning: trimmedOutput)
                    }
                } catch {
                    let errorLog = "\n[ERROR] \(error.localizedDescription)\n"
                    if let handle = FileHandle(forWritingAtPath: logFile) {
                        handle.seekToEndOfFile()
                        handle.write(errorLog.data(using: .utf8)!)
                        handle.closeFile()
                    }
                    continuation.resume(throwing: LTXError.generationFailed(error.localizedDescription))
                }
            }
        }
    }
}
