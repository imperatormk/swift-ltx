// Lightweight tensor type — just shape + Metal buffer.
// No autograd, no lazy eval. Direct GPU memory.

import Metal

public struct Tensor {
    public let buffer: MTLBuffer
    public let shape: [Int]
    public let dtype: TensorDType

    public var count: Int { shape.reduce(1, *) }
    public var byteCount: Int { count * dtype.size }

    /// Dimension at index (supports negative indexing).
    public func dim(_ i: Int) -> Int {
        let idx = i >= 0 ? i : shape.count + i
        return shape[idx]
    }

    /// Create from existing Metal buffer.
    public init(buffer: MTLBuffer, shape: [Int], dtype: TensorDType = .float32) {
        self.buffer = buffer
        self.shape = shape
        self.dtype = dtype
    }

    /// Create from existing Metal buffer with byte offset (copies the slice, uses pool).
    public init(buffer: MTLBuffer, shape: [Int], offset: Int, dtype: TensorDType = .float32) {
        let count = shape.reduce(1, *)
        let bytes = count * 4  // float32
        let buf = MetalContext.shared.bufferPool.get(length: bytes)
        memcpy(buf.contents(), buffer.contents() + offset, bytes)
        self.buffer = buf
        self.shape = shape
        self.dtype = dtype
    }

    /// Create from Float array.
    public init(_ data: [Float], shape: [Int]) {
        let ctx = MetalContext.shared
        self.buffer = data.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        self.shape = shape
        self.dtype = .float32
    }

    /// Create uninitialized tensor from pool (for ops that fully overwrite output).
    public static func empty(_ shape: [Int], dtype: TensorDType = .float32) -> Tensor {
        let count = shape.reduce(1, *)
        let bytes: Int
        switch dtype {
        case .float32: bytes = count * 4
        case .float16, .bfloat16: bytes = count * 2
        case .int8, .uint8: bytes = count
        case .int32, .uint32: bytes = count * 4
        }
        let buf = MetalContext.shared.bufferPool.get(length: bytes)
        return Tensor(buffer: buf, shape: shape, dtype: dtype)
    }

    /// Create zeroed tensor (uses buffer pool for reuse).
    public static func zeros(_ shape: [Int], dtype: TensorDType = .float32) -> Tensor {
        let count = shape.reduce(1, *)
        let bytes: Int
        switch dtype {
        case .float32: bytes = count * 4
        case .float16, .bfloat16: bytes = count * 2
        case .int8, .uint8: bytes = count
        case .int32, .uint32: bytes = count * 4
        }
        let buf = MetalContext.shared.bufferPool.get(length: bytes)
        memset(buf.contents(), 0, bytes)
        return Tensor(buffer: buf, shape: shape, dtype: dtype)
    }

    /// Read back float data from GPU.
    public func toFloats() -> [Float] {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
