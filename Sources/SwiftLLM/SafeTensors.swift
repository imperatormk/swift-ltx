// SafeTensors loader — mmap-based, zero-copy weight loading.
// Format: 8-byte header length (LE u64) + JSON header + raw tensor data

import Foundation

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
    private let dataOffset: Int // where tensor data starts

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

/// Load all safetensors shards from a model directory.
public func loadSafeTensors(from directory: URL) throws -> [SafeTensorsFile] {
    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    return try files.map { try SafeTensorsFile(url: $0) }
}
