import SwiftUI
import Metal

struct ParametersView: View {
    @EnvironmentObject var presetManager: PresetManager
    @AppStorage(LTXModelCatalog.selectedModelIDKey) private var selectedModelID = LTXModelCatalog.defaultModelID
    
    @Binding var parameters: GenerationParameters
    @State private var showSavePreset = false
    @State private var newPresetName = ""
    @State private var availableVRAM = getAvailableVRAM()

    private var selectedModel: LTXModel {
        LTXModelCatalog.resolvedModel(id: selectedModelID)
    }
    
    let vramTimer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Preset picker
            VStack(alignment: .leading, spacing: 8) {
                Label("Preset", systemImage: "slider.horizontal.3")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Picker("", selection: $presetManager.selectedPreset) {
                        ForEach(presetManager.presets) { preset in
                            Text(preset.name).tag(preset as Preset?)
                        }
                    }
                    .labelsHidden()
                    .onChange(of: presetManager.selectedPreset) { _, newValue in
                        if let preset = newValue {
                            parameters = preset.parameters
                        }
                    }
                    
                    Button {
                        showSavePreset = true
                    } label: {
                        Image(systemName: "plus.circle")
                    }
                    .buttonStyle(.borderless)
                    .help("Save current settings as preset")
                }
            }
            
            Divider()
            
            // Parameters
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let tips = selectedModel.tips {
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "info.circle.fill")
                                .foregroundStyle(.blue)
                            Text(tips)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.blue.opacity(0.08))
                        )
                    }

                    if let warning = selectedModel.qualityWarning {
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                            Text(warning)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.orange.opacity(0.08))
                        )
                    }

                    // Inference steps
                    ParameterSlider(
                        title: "Inference Steps",
                        value: Binding(
                            get: { Double(parameters.numInferenceSteps) },
                            set: { parameters.numInferenceSteps = Int($0) }
                        ),
                        range: 10...100,
                        step: 5,
                        icon: "arrow.triangle.2.circlepath"
                    )

                    if let range = selectedModel.recommendedSteps {
                        HStack(spacing: 4) {
                            Image(systemName: "target")
                                .foregroundStyle(.blue)
                            Text("Recommended: \(range.lowerBound)–\(range.upperBound) steps")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    // Guidance scale
                    ParameterSlider(
                        title: "Guidance Scale",
                        value: $parameters.guidanceScale,
                        range: 1...15,
                        step: 0.5,
                        icon: "dial.medium",
                        format: "%.1f"
                    )
                    
                    Divider()
                    
                    // Resolution
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Resolution", systemImage: "rectangle.dashed")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        ResolutionSlider(
                            title: "Width",
                            value: $parameters.width,
                            range: 256...2560,
                            step: 64,
                            icon: "arrow.left.and.right"
                        )

                        ResolutionSlider(
                            title: "Height",
                            value: $parameters.height,
                            range: 256...2560,
                            step: 64,
                            icon: "arrow.up.and.down"
                        )

                        HStack(spacing: 12) {
                            AspectPreview(width: parameters.width, height: parameters.height)

                            VStack(alignment: .leading, spacing: 4) {
                                Text("\\(parameters.width)×\\(parameters.height)")
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)

                                Text(aspectRatioText(width: parameters.width, height: parameters.height))
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()
                        }
                        .padding(.top, 4)

                        if parameters.width * parameters.height > 768 * 512 {
                            HStack(alignment: .top, spacing: 6) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                                Text("Higher resolutions significantly increase generation time and memory usage. LTX-2 models are trained at 768×512; larger outputs may not improve quality and can cause Metal OOM errors. Reduce frame count or use aggressive VAE tiling if you hit memory limits.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.top, 4)
                        }
                    }
                    
                    Divider()
                    
                    // Frame count
                    ParameterSlider(
                        title: "Frames",
                        value: Binding(
                            get: { Double(parameters.numFrames) },
                            set: { parameters.numFrames = Int($0) }
                        ),
                        range: 25...1000,
                        step: 25,
                        icon: "film.stack"
                    )
                    
                    if parameters.numFrames > 500 {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                            Text("High frame count may exceed GPU memory")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    // FPS
                    VStack(alignment: .leading, spacing: 8) {
                        Label("FPS", systemImage: "speedometer")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        
                        Picker("", selection: $parameters.fps) {
                            Text("12 fps").tag(12)
                            Text("20 fps").tag(20)
                            Text("24 fps").tag(24)
                            Text("30 fps").tag(30)
                        }
                        .pickerStyle(.segmented)

                        Text("Synchronized speech works best at 24 fps.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)

                        if parameters.fps != 24 {
                            HStack(spacing: 6) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                                Text("Non-24 fps can reduce speech/lip-sync quality.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    
                    // Video length estimate
                    HStack {
                        Image(systemName: "film")
                        Text("Video Length: \(parameters.videoLength)")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    
                    Divider()
                    
                    // Seed
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Seed", systemImage: "dice")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        
                        HStack {
                            TextField("Random", value: $parameters.seed, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 120)
                            
                            Button {
                                parameters.seed = Int.random(in: 0..<Int(Int32.max))
                            } label: {
                                Image(systemName: "dice.fill")
                            }
                            .buttonStyle(.borderless)
                            .help("Generate random seed")
                            
                            if parameters.seed != nil {
                                Button {
                                    parameters.seed = nil
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                }
                                .buttonStyle(.borderless)
                                .help("Clear seed (use random)")
                            }
                        }
                    }
                    
                    Divider()
                    
                    // VAE Tiling Mode
                    VStack(alignment: .leading, spacing: 8) {
                        Label("VAE Tiling", systemImage: "square.grid.3x3")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        
                        Picker("", selection: $parameters.vaeTilingMode) {
                            Text("Auto").tag("auto")
                            Text("None").tag("none")
                            Text("Default").tag("default")
                            Text("Aggressive").tag("aggressive")
                            Text("Conservative").tag("conservative")
                            Text("Spatial Only").tag("spatial")
                            Text("Temporal Only").tag("temporal")
                        }
                        .labelsHidden()
                        
                        Text("Controls memory vs speed tradeoff during decoding. Aggressive reduces peak memory but can trigger macOS Metal “Impacting Interactivity” watchdogs on small resolutions; use Auto or Conservative if that happens.")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    
                    // Estimated time
                    HStack {
                        Image(systemName: "clock")
                        Text("Estimated: \(parameters.estimatedDuration)")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                    
                    Spacer()
                    
                    // VRAM info
                    HStack {
                        Spacer()
                        HStack(spacing: 4) {
                            Image(systemName: "memorychip")
                            Text("\(availableVRAM) available")
                        }
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .onReceive(vramTimer) { _ in
                            availableVRAM = getAvailableVRAM()
                        }
                    }
                }
                .padding(.horizontal, 4)
            }
        }
        .padding()
        .sheet(isPresented: $showSavePreset) {
            SavePresetSheet(
                presetName: $newPresetName,
                parameters: parameters,
                isPresented: $showSavePreset
            )
            .environmentObject(presetManager)
        }
    }
}

struct ParameterSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let icon: String
    var format: String = "%.0f"
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(title, systemImage: icon)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                
                Spacer()
                
                Text(String(format: format, value))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
            }
            
            Slider(value: $value, in: range, step: step)
        }
    }
}

struct ResolutionSlider: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int
    let icon: String

    @State private var editText: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(title, systemImage: icon)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Spacer()

                TextField("", text: $editText)
                    .textFieldStyle(.plain)
                    .multilineTextAlignment(.trailing)
                    .font(.caption.monospaced())
                    .frame(width: 56)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
                    .onSubmit {
                        commitEdit()
                    }
                    .onChange(of: value) { _, newValue in
                        editText = String(newValue)
                    }
                    .onAppear {
                        editText = String(value)
                    }
            }

            Slider(
                value: Binding(
                    get: { Double(value) },
                    set: { value = Int($0) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound),
                step: Double(step)
            )
        }
    }

    private func commitEdit() {
        guard let parsed = Int(editText.trimmingCharacters(in: .whitespaces)) else {
            editText = String(value)
            return
        }
        // Clamp to range, then round down to nearest multiple of step.
        let clamped = min(max(parsed, range.lowerBound), range.upperBound)
        value = (clamped / step) * step
        editText = String(value)
    }
}

struct AspectPreview: View {
    let width: Int
    let height: Int

    var body: some View {
        let box: CGFloat = 120
        let ratio = CGFloat(width) / CGFloat(height)
        let rectWidth: CGFloat
        let rectHeight: CGFloat
        if ratio >= 1 {
            rectWidth = box
            rectHeight = box / ratio
        } else {
            rectWidth = box * ratio
            rectHeight = box
        }

        return ZStack {
            RoundedRectangle(cornerRadius: 4)
                .strokeBorder(Color.secondary.opacity(0.5), lineWidth: 1)
                .frame(width: box, height: box)
            RoundedRectangle(cornerRadius: 3)
                .fill(Color.accentColor.opacity(0.35))
                .frame(width: rectWidth, height: rectHeight)
        }
    }
}

private func aspectRatioText(width: Int, height: Int) -> String {
    guard width > 0, height > 0 else { return "" }
    let ratio = Double(width) / Double(height)
    // Common aspect ratios (w:h)
    let common: [(String, Double)] = [
        ("21:9", 21.0 / 9.0), ("16:9", 16.0 / 9.0), ("3:2", 3.0 / 2.0),
        ("4:3", 4.0 / 3.0), ("1:1", 1.0), ("3:4", 3.0 / 4.0),
        ("2:3", 2.0 / 3.0), ("9:16", 9.0 / 16.0), ("9:21", 9.0 / 21.0)
    ]
    if let best = common.min(by: { abs($0.1 - ratio) < abs($1.1 - ratio) }),
       abs(best.1 - ratio) < 0.05 {
        return "\(best.0) (\(String(format: "%.2f", ratio)))"
    }
    // Fallback: GCD reduction
    let gcd = greatestCommonDivisor(width, height)
    return "\(width / gcd):\(height / gcd) (\(String(format: "%.2f", ratio)))"
}

private func greatestCommonDivisor(_ a: Int, _ b: Int) -> Int {
    var x = a
    var y = b
    while y != 0 {
        let t = y
        y = x % y
        x = t
    }
    return x
}

struct SavePresetSheet: View {
    @EnvironmentObject var presetManager: PresetManager
    @Binding var presetName: String
    let parameters: GenerationParameters
    @Binding var isPresented: Bool
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Save Preset")
                .font(.headline)
            
            TextField("Preset Name", text: $presetName)
                .textFieldStyle(.roundedBorder)
            
            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)
                
                Spacer()
                
                Button("Save") {
                    _ = presetManager.saveCurrentAsPreset(name: presetName, parameters: parameters)
                    presetName = ""
                    isPresented = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(presetName.isEmpty)
            }
        }
        .padding()
        .frame(width: 300)
    }
}

private func getAvailableVRAM() -> String {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return "N/A"
    }
    
    // On Apple Silicon, recommendedMaxWorkingSetSize gives us usable memory
    let bytes = device.recommendedMaxWorkingSetSize
    let gb = Double(bytes) / 1_073_741_824.0
    return String(format: "%.0fGB", gb)
}

#Preview {
    ParametersView(parameters: .constant(.default))
        .environmentObject(PresetManager())
        .frame(width: 300, height: 600)
}
