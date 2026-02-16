import SwiftUI
import SwiftLLM

struct VideoView: View {
    @StateObject private var runner = VideoRunner()
    @State private var showDebug = true
    @State private var didAutoStart = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        // Device info
                        HStack(spacing: 12) {
                            Image(systemName: "film.stack")
                                .font(.title2)
                                .foregroundStyle(.purple)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("GPU")
                                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.gray)
                                Text(runner.deviceName)
                                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.white)
                            }
                            Spacer()
                        }
                        .padding(14)
                        .background(Color.white.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 10))

                        // Paths
                        #if os(macOS)
                        pathField("DiT model dir", text: $runner.ditPath)
                        pathField("T5 embeddings", text: $runner.embeddingsPath)
                        pathField("VAE weights", text: $runner.vaePath)
                        #endif

                        // Generation params
                        HStack(spacing: 16) {
                            paramField("Frames", value: Binding(
                                get: { runner.numFrames },
                                set: { runner.numFrames = $0 }
                            ), range: 1...32)
                        }
                        HStack(spacing: 16) {
                            paramField("H", value: Binding(
                                get: { runner.height },
                                set: { runner.height = $0 }
                            ), range: 8...256)
                            paramField("W", value: Binding(
                                get: { runner.width },
                                set: { runner.width = $0 }
                            ), range: 8...256)
                        }
                        HStack(spacing: 16) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Seed")
                                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.gray)
                                TextField("42", text: Binding(
                                    get: { String(runner.seed) },
                                    set: { runner.seed = UInt64($0) ?? 42 }
                                ))
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                            }
                            Spacer()
                        }

                        // Debug toggle
                        Toggle(isOn: $showDebug) {
                            HStack(spacing: 6) {
                                Image(systemName: "ladybug")
                                    .foregroundStyle(.yellow)
                                Text("Debug Log")
                                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.white)
                            }
                        }
                        .tint(.yellow)

                        // Status
                        Text(runner.statusText)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(.gray)
                            .textSelection(.enabled)

                        // Progress
                        if let (step, total) = runner.progress {
                            ProgressView(value: Double(step), total: Double(total))
                                .tint(.purple)
                        }

                        // Debug log
                        if showDebug, !runner.debugLog.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("DEBUG")
                                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                                        .foregroundStyle(.yellow)
                                    Spacer()
                                    Button(action: {
                                        #if os(macOS)
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(runner.debugLog, forType: .string)
                                        #endif
                                    }) {
                                        Image(systemName: "doc.on.doc")
                                            .font(.system(size: 12))
                                            .foregroundStyle(.yellow.opacity(0.6))
                                    }
                                    .buttonStyle(.plain)
                                }
                                SelectableText(runner.debugLog)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .padding(10)
                            .background(Color.yellow.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }

                        // Decoded image
                        if let frame = runner.decodedImage {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("OUTPUT — \(frame.width)x\(frame.height)")
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundStyle(.green)

                                if let img = videoFrameToImage(frame) {
                                    #if os(macOS)
                                    Image(nsImage: img)
                                        .resizable()
                                        .interpolation(.none)
                                        .aspectRatio(contentMode: .fit)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                    #else
                                    Image(uiImage: img)
                                        .resizable()
                                        .interpolation(.none)
                                        .aspectRatio(contentMode: .fit)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                    #endif
                                }
                            }
                            .padding(12)
                            .background(Color.white.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }

                        Spacer(minLength: 20)
                    }
                    .padding(20)
                }

                Divider().overlay(Color.white.opacity(0.1))

                // Buttons
                HStack(spacing: 10) {
                    Spacer()
                    if runner.isRunning {
                        ProgressView()
                            .controlSize(.small)
                        Text("Working...")
                            .font(.system(size: 14, weight: .medium, design: .monospaced))
                            .foregroundStyle(.gray)
                    } else {
                        Button(action: { runner.generate() }) {
                            HStack {
                                Image(systemName: "play.fill")
                                Text("Generate")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.purple)

                        Button(action: { runner.generateFromMLXAdaIN() }) {
                            HStack {
                                Image(systemName: "arrow.right.circle.fill")
                                Text("MLX P2")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                    }
                    Spacer()
                }
                .padding(16)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(red: 0.08, green: 0.08, blue: 0.10))
            .navigationTitle("Video Gen")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .onAppear {
                #if os(macOS)
                if !didAutoStart {
                    didAutoStart = true
                    runner.generate()
                }
                #endif
            }
        }
    }

    #if os(macOS)
    private func pathField(_ label: String, text: Binding<String>) -> some View {
        HStack(spacing: 8) {
            TextField(label, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, design: .monospaced))
                .disabled(runner.isRunning)
        }
    }
    #endif

    private func paramField(_ label: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.gray)
            TextField(label, text: Binding(
                get: { String(value.wrappedValue) },
                set: { value.wrappedValue = max(range.lowerBound, min(range.upperBound, Int($0) ?? value.wrappedValue)) }
            ))
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 12, design: .monospaced))
            .disabled(runner.isRunning)
        }
    }
}

// MARK: - Selectable text

#if os(macOS)
import AppKit

struct SelectableText: NSViewRepresentable {
    let text: String
    init(_ text: String) { self.text = text }

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField(wrappingLabelWithString: "")
        field.isEditable = false
        field.isSelectable = true
        field.backgroundColor = .clear
        field.isBordered = false
        field.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        field.textColor = NSColor.systemYellow.withAlphaComponent(0.8)
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
#else
struct SelectableText: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .font(.system(size: 10, weight: .regular, design: .monospaced))
            .foregroundStyle(.yellow.opacity(0.8))
            .textSelection(.enabled)
    }
}
#endif

// MARK: - Image conversion

#if os(macOS)
private func videoFrameToImage(_ frame: DecodedFrame) -> NSImage? {
    let w = frame.width, h = frame.height
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(
              data: UnsafeMutableRawPointer(mutating: (frame.rgbaData as NSData).bytes),
              width: w, height: h,
              bitsPerComponent: 8, bytesPerRow: w * 4,
              space: colorSpace,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          ),
          let cgImage = ctx.makeImage()
    else { return nil }
    return NSImage(cgImage: cgImage, size: NSSize(width: w, height: h))
}
#else
import UIKit
private func videoFrameToImage(_ frame: DecodedFrame) -> UIImage? {
    let w = frame.width, h = frame.height
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(
              data: UnsafeMutableRawPointer(mutating: (frame.rgbaData as NSData).bytes),
              width: w, height: h,
              bitsPerComponent: 8, bytesPerRow: w * 4,
              space: colorSpace,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          ),
          let cgImage = ctx.makeImage()
    else { return nil }
    return UIImage(cgImage: cgImage)
}
#endif

#Preview {
    VideoView()
        .preferredColorScheme(.dark)
}
