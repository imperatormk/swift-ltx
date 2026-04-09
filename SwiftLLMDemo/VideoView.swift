import SwiftUI

struct VideoView: View {
    @StateObject private var model = VideoGenerationViewModel()
    @State private var showDebugLog = true
    @State private var currentFrameIndex = 0
    @State private var frameTimer: Timer?
    @State private var isPlaying = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        headerCard
                        modelCard
                        settingsCard
                        outputCard
                    }
                    .padding(20)
                }

                Divider().overlay(Color.white.opacity(0.08))
                footerBar
                    .padding(16)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(red: 0.07, green: 0.08, blue: 0.09))
            .navigationTitle("LTX Video")
        }
    }

    private var headerCard: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("GPU")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.gray)
                Text(model.deviceName)
                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text("PEAK MEM")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.gray)
                Text(model.peakMemText)
                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.orange)
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Model Files")
            pathField("DiT", text: $model.ditPath)
            pathField("VAE", text: $model.vaePath)
            pathField("Upsampler", text: $model.upsamplerPath)
            Toggle("Encode prompt with T5", isOn: $model.useT5Encoder)
                .font(.system(size: 13, design: .monospaced))
                .tint(.orange)
                .disabled(model.isRunning)

            if model.useT5Encoder {
                pathField("T5 weights", text: $model.t5WeightsPath)
                pathField("T5 tokenizer", text: $model.t5TokenizerPath)
                VStack(alignment: .leading, spacing: 6) {
                    Text("Prompt")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.gray)
                    TextField("Prompt", text: $model.customPrompt, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 13, design: .monospaced))
                        .disabled(model.isRunning)
                }
            } else {
                pathField("T5 embeddings", text: $model.embeddingsPath)
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var settingsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Generation")
            HStack(spacing: 12) {
                numericField("Frames", value: $model.numFrames, range: 1...32)
                numericField("Height", value: $model.height, range: 8...256)
                numericField("Width", value: $model.width, range: 8...256)
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Seed")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.gray)
                TextField(
                    "Seed",
                    text: Binding(
                        get: { String(model.seed) },
                        set: { model.seed = UInt64($0) ?? model.seed }
                    )
                )
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 13, design: .monospaced))
                .disabled(model.isRunning)
            }

            if let progress = model.progress {
                VStack(alignment: .leading, spacing: 6) {
                    Text("\(progress.label) \(progress.step)/\(progress.total)")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.gray)
                    ProgressView(value: Double(progress.step), total: Double(progress.total))
                        .tint(.orange)
                }
            }

            Text(model.statusText)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(.gray)
                .textSelection(.enabled)

            Toggle("Show debug log", isOn: $showDebugLog)
                .font(.system(size: 13, design: .monospaced))
                .tint(.yellow)

            if showDebugLog, !model.debugLog.isEmpty {
                SelectableText(model.debugLog)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(Color.yellow.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private var outputCard: some View {
        if !model.decodedFrames.isEmpty {
            let frames = model.decodedFrames
            let frame = frames[currentFrameIndex % frames.count]
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    sectionTitle("Output")
                    Spacer()
                    Text("F\(currentFrameIndex % frames.count + 1)/\(frames.count)")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.gray)
                }

                if let image = videoFrameToImage(frame) {
                    #if os(macOS)
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.high)
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    #else
                    Image(uiImage: image)
                        .resizable()
                        .interpolation(.high)
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    #endif
                }

                HStack(spacing: 16) {
                    Button(action: togglePlayback) {
                        Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.orange)

                    Slider(
                        value: Binding(
                            get: { Double(currentFrameIndex % frames.count) },
                            set: { currentFrameIndex = Int($0) }
                        ),
                        in: 0...Double(frames.count - 1),
                        step: 1
                    )
                    .tint(.orange)

                    Button(action: { exportMP4(frames: frames) }) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.orange)
                }
            }
            .padding(14)
            .background(Color.white.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .onAppear { startPlayback() }
            .onDisappear { stopPlayback() }
        }
    }

    private var footerBar: some View {
        HStack(spacing: 12) {
            if model.isRunning {
                Button("Cancel") {
                    model.cancel()
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            } else {
                Button("Generate") {
                    currentFrameIndex = 0
                    model.generate()
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
            }
            Spacer()
        }
    }

    private func sectionTitle(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 11, weight: .bold, design: .monospaced))
            .foregroundStyle(.white.opacity(0.75))
    }

    private func pathField(_ label: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.gray)
            TextField(label, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, design: .monospaced))
                .disabled(model.isRunning)
        }
    }

    private func numericField(_ label: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.gray)
            TextField(
                label,
                text: Binding(
                    get: { String(value.wrappedValue) },
                    set: { value.wrappedValue = max(range.lowerBound, min(range.upperBound, Int($0) ?? value.wrappedValue)) }
                )
            )
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 12, design: .monospaced))
            .disabled(model.isRunning)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func togglePlayback() {
        if isPlaying {
            stopPlayback()
        } else {
            startPlayback()
        }
    }

    private func startPlayback() {
        guard model.decodedFrames.count > 1 else { return }
        stopPlayback()
        isPlaying = true
        frameTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 24.0, repeats: true) { _ in
            DispatchQueue.main.async {
                currentFrameIndex = (currentFrameIndex + 1) % model.decodedFrames.count
            }
        }
    }

    private func stopPlayback() {
        frameTimer?.invalidate()
        frameTimer = nil
        isPlaying = false
    }

    private func exportMP4(frames: [DecodedFrame]) {
        guard let first = frames.first, first.width > 0, first.height > 0 else { return }
        #if os(macOS)
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = "output.mp4"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            writeMP4(frames: frames, to: url, fps: 24)
        }
        #endif
    }
}

import AVFoundation

private func writeMP4(frames: [DecodedFrame], to url: URL, fps: Int) {
    guard let first = frames.first else { return }
    let width = first.width
    let height = first.height

    try? FileManager.default.removeItem(at: url)
    guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return }

    let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height
        ]
    )
    input.expectsMediaDataInRealTime = false
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    for (index, frame) in frames.enumerated() {
        while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.01) }
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard let pixelBuffer else { continue }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let destination = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        frame.rgbaData.withUnsafeBytes { source in
            let sourceBytes = source.assumingMemoryBound(to: UInt8.self).baseAddress!
            for y in 0..<height {
                for x in 0..<width {
                    let sourceIndex = (y * width + x) * 4
                    let destinationIndex = y * bytesPerRow + x * 4
                    destination[destinationIndex + 0] = sourceBytes[sourceIndex + 2]
                    destination[destinationIndex + 1] = sourceBytes[sourceIndex + 1]
                    destination[destinationIndex + 2] = sourceBytes[sourceIndex + 0]
                    destination[destinationIndex + 3] = sourceBytes[sourceIndex + 3]
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        adaptor.append(pixelBuffer, withPresentationTime: CMTime(value: Int64(index), timescale: Int32(fps)))
    }

    input.markAsFinished()
    writer.finishWriting {}
}

#if os(macOS)
import AppKit

struct SelectableText: NSViewRepresentable {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField(wrappingLabelWithString: "")
        field.isEditable = false
        field.isSelectable = true
        field.backgroundColor = .clear
        field.isBordered = false
        field.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        field.textColor = NSColor.systemYellow.withAlphaComponent(0.85)
        field.lineBreakMode = .byCharWrapping
        field.maximumNumberOfLines = 0
        field.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return field
    }

    func updateNSView(_ field: NSTextField, context: Context) {
        if field.stringValue != text {
            field.stringValue = text
        }
    }
}

private func videoFrameToImage(_ frame: DecodedFrame) -> NSImage? {
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let context = CGContext(
              data: UnsafeMutableRawPointer(mutating: (frame.rgbaData as NSData).bytes),
              width: frame.width,
              height: frame.height,
              bitsPerComponent: 8,
              bytesPerRow: frame.width * 4,
              space: colorSpace,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          ),
          let image = context.makeImage()
    else {
        return nil
    }
    return NSImage(cgImage: image, size: NSSize(width: frame.width, height: frame.height))
}
#else
import UIKit

struct SelectableText: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(.yellow.opacity(0.85))
            .textSelection(.enabled)
    }
}

private func videoFrameToImage(_ frame: DecodedFrame) -> UIImage? {
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let context = CGContext(
              data: UnsafeMutableRawPointer(mutating: (frame.rgbaData as NSData).bytes),
              width: frame.width,
              height: frame.height,
              bitsPerComponent: 8,
              bytesPerRow: frame.width * 4,
              space: colorSpace,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          ),
          let image = context.makeImage()
    else {
        return nil
    }
    return UIImage(cgImage: image)
}
#endif

#Preview {
    VideoView()
        .preferredColorScheme(.dark)
}
