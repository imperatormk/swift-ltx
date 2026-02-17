// SafeTensors loader — mmap-based, zero-copy weight loading.
// Format: 8-byte header length (LE u64) + JSON header + raw tensor data

import Foundation
import Metal

public struct TensorInfo {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int]
    public let offset: Int      // byte offset into data region
    public let byteCount: Int   // total bytes for this tensor
}

public enum TensorDType: String {
    case float32 = "F32"
    case float16 = "F16"
    case bfloat16 = "BF16"
    case int8 = "I8"
    case uint8 = "U8"
    case int32 = "I32"
    case uint32 = "U32"

    public var size: Int {
        switch self {
        case .float32, .int32, .uint32: return 4
        case .float16, .bfloat16: return 2
        case .int8, .uint8: return 1
        }
    }
}

/// Memory-mapped safetensors file. Tensors are accessible as raw pointers
/// without copying — ideal for feeding directly into Metal buffers.
public class SafeTensorsFile {
    public let tensors: [String: TensorInfo]
    private let fileData: Data  // mmap'd
    public let dataOffset: Int // where tensor data starts

    public init(url: URL) throws {
        // mmap the file
        self.fileData = try Data(contentsOf: url, options: .mappedIfSafe)

        // Read header length (first 8 bytes, little-endian u64)
        let headerLen = fileData.withUnsafeBytes { ptr in
            ptr.load(as: UInt64.self)
        }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerLen)
        self.dataOffset = headerEnd

        // Parse JSON header
        let headerData = fileData.subdata(in: headerStart..<headerEnd)
        let json = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]

        var tensors: [String: TensorInfo] = [:]
        for (key, value) in json {
            guard key != "__metadata__",
                  let info = value as? [String: Any],
                  let dtypeStr = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }

            let dtype: TensorDType
            switch dtypeStr {
            case "F32": dtype = .float32
            case "F16": dtype = .float16
            case "BF16": dtype = .bfloat16
            case "I8": dtype = .int8
            case "U8": dtype = .uint8
            case "I32": dtype = .int32
            case "U32": dtype = .uint32
            default: continue
            }

            tensors[key] = TensorInfo(
                name: key,
                dtype: dtype,
                shape: shape,
                offset: offsets[0],
                byteCount: offsets[1] - offsets[0]
            )
        }
        self.tensors = tensors
    }

    /// Get a raw pointer to tensor data. Zero-copy via mmap.
    /// NOTE: The pointer is only valid while the SafeTensorsFile is alive (mmap-backed).
    public func pointer(for name: String) -> UnsafeRawPointer? {
        guard let info = tensors[name] else { return nil }
        // fileData is mmap'd and retained by self, so the pointer remains valid
        // as long as self is alive. We use withUnsafeBytes to get the base, then
        // compute the offset. The mmap backing keeps the memory valid.
        return fileData.withUnsafeBytes { ptr in
            ptr.baseAddress! + dataOffset + info.offset
        }
    }

    /// Access tensor data safely within a closure where the pointer is guaranteed valid.
    public func withPointer<R>(for name: String, body: (UnsafeRawPointer, TensorInfo) throws -> R) rethrows -> R? {
        guard let info = tensors[name] else { return nil }
        return try fileData.withUnsafeBytes { ptr in
            let p = ptr.baseAddress! + dataOffset + info.offset
            return try body(p, info)
        }
    }

    /// Get tensor data as a typed buffer.
    public func buffer<T>(for name: String, as type: T.Type) -> UnsafeBufferPointer<T>? {
        guard let info = tensors[name] else { return nil }
        return fileData.withUnsafeBytes { ptr in
            let start = (ptr.baseAddress! + dataOffset + info.offset)
                .assumingMemoryBound(to: T.self)
            return UnsafeBufferPointer(start: start, count: info.byteCount / MemoryLayout<T>.size)
        }
    }
}

/// Lightweight index for streaming tensor reads without keeping the file mmap'd.
/// Parses header once, then reads individual tensors via FileHandle (no resident pages).
public class SafeTensorsIndex {
    public let tensors: [String: TensorInfo]
    public let url: URL
    public let dataOffset: Int

    /// Create from an already-parsed SafeTensorsFile (reuses header parse).
    public init(from file: SafeTensorsFile, url: URL) {
        self.tensors = file.tensors
        self.url = url
        self.dataOffset = file.dataOffset
    }

    /// Parse header only (small read), don't mmap the whole file.
    public init(url: URL) throws {
        self.url = url
        let fh = try FileHandle(forReadingFrom: url)
        defer { fh.closeFile() }
        let headerLenData = fh.readData(ofLength: 8)
        let headerLen = headerLenData.withUnsafeBytes { $0.load(as: UInt64.self) }
        self.dataOffset = 8 + Int(headerLen)
        let headerData = fh.readData(ofLength: Int(headerLen))
        let json = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]
        var tensors: [String: TensorInfo] = [:]
        for (key, value) in json {
            guard key != "__metadata__",
                  let info = value as? [String: Any],
                  let dtypeStr = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }
            let dtype: TensorDType
            switch dtypeStr {
            case "F32": dtype = .float32
            case "F16": dtype = .float16
            case "BF16": dtype = .bfloat16
            case "I8": dtype = .int8
            case "U8": dtype = .uint8
            case "I32": dtype = .int32
            case "U32": dtype = .uint32
            default: continue
            }
            tensors[key] = TensorInfo(name: key, dtype: dtype, shape: shape,
                                       offset: offsets[0], byteCount: offsets[1] - offsets[0])
        }
        self.tensors = tensors
    }

    /// Read a tensor's raw bytes from disk (no mmap, no resident pages after use).
    public func readTensorData(_ name: String) -> Data? {
        guard let info = tensors[name] else { return nil }
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { fh.closeFile() }
        fh.seek(toFileOffset: UInt64(dataOffset + info.offset))
        return fh.readData(ofLength: info.byteCount)
    }

    /// Read tensor directly into a Metal buffer — zero intermediate Data allocation.
    /// Uses POSIX read() into the buffer's contents pointer.
    public func readTensorIntoBuffer(_ name: String) -> MTLBuffer? {
        guard let info = tensors[name] else { return nil }
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        let buf = MetalContext.shared.device.makeBuffer(length: info.byteCount, options: .storageModeShared)!
        _posixRead(fd: fd, offset: dataOffset + info.offset, dst: buf.contents(), count: info.byteCount)
        return buf
    }

    /// Read tensor into an existing destination pointer (for conversion workflows).
    /// Caller provides pre-allocated memory — no Metal buffer or Data allocated.
    public func readTensorRaw(_ name: String, into dst: UnsafeMutableRawPointer) -> Bool {
        guard let info = tensors[name] else { return false }
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        _posixRead(fd: fd, offset: dataOffset + info.offset, dst: dst, count: info.byteCount)
        return true
    }
}

private func _posixRead(fd: Int32, offset: Int, dst: UnsafeMutableRawPointer, count: Int) {
    lseek(fd, off_t(offset), SEEK_SET)
    var remaining = count
    var ptr = dst
    while remaining > 0 {
        let n = read(fd, ptr, remaining)
        if n <= 0 { break }
        remaining -= n
        ptr += n
    }
}

/// Write a single tensor to safetensors format.
public func saveSafetensors(name: String, data: UnsafeRawPointer, byteCount: Int,
                            dtype: TensorDType, shape: [Int], to url: URL) throws {
    // Build JSON header
    let shapeStr = "[" + shape.map { String($0) }.joined(separator: ",") + "]"
    let header = "{\"\(name)\":{\"dtype\":\"\(dtype.rawValue)\",\"shape\":\(shapeStr),\"data_offsets\":[0,\(byteCount)]}}"
    let headerData = header.data(using: .utf8)!
    // Pad header to 8-byte alignment
    let paddedLen = (headerData.count + 7) & ~7
    var out = Data(count: 8 + paddedLen + byteCount)
    out.withUnsafeMutableBytes { ptr in
        // 8-byte LE header length
        ptr.storeBytes(of: UInt64(paddedLen), as: UInt64.self)
        // Header JSON + space padding
        (ptr.baseAddress! + 8).copyMemory(from: (headerData as NSData).bytes, byteCount: headerData.count)
        for i in headerData.count..<paddedLen {
            (ptr.baseAddress! + 8 + i).storeBytes(of: UInt8(0x20), as: UInt8.self) // space padding
        }
        // Tensor data
        (ptr.baseAddress! + 8 + paddedLen).copyMemory(from: data, byteCount: byteCount)
    }
    try out.write(to: url)
}

/// Write a Metal buffer tensor to safetensors.
public func saveSafetensors(name: String, buffer: MTLBuffer, dtype: TensorDType, shape: [Int], to url: URL) throws {
    let byteCount = shape.reduce(1, *) * dtype.size
    try saveSafetensors(name: name, data: buffer.contents(), byteCount: byteCount, dtype: dtype, shape: shape, to: url)
}

/// Load all safetensors shards from a model directory.
public func loadSafeTensors(from directory: URL) throws -> [SafeTensorsFile] {
    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    return try files.map { try SafeTensorsFile(url: $0) }
}
