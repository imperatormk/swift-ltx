// Metal device + command queue singleton.

import Metal

public final class MetalContext: @unchecked Sendable {
    nonisolated(unsafe) public static let shared = MetalContext()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let bufferPool = BufferPool()

    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
    }

    /// Create a buffer from raw bytes (shared storage for macOS+iOS).
    public func makeBuffer(_ data: UnsafeRawPointer, length: Int) -> MTLBuffer {
        device.makeBuffer(bytes: data, length: length, options: .storageModeShared)!
    }

    /// Create a pooled buffer from raw bytes (recycled between passes).
    public func makePooledBuffer(_ data: UnsafeRawPointer, length: Int) -> MTLBuffer {
        let buf = bufferPool.get(length: length)
        memcpy(buf.contents(), data, length)
        return buf
    }

    /// Create an empty buffer.
    public func makeBuffer(length: Int) -> MTLBuffer {
        device.makeBuffer(length: length, options: .storageModeShared)!
    }

    /// Run a compute pass synchronously.
    public func run(_ body: (MTLComputeCommandEncoder) -> Void) {
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        body(enc)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }
}

// MARK: - Buffer Pool (arena-style reuse)

/// Reuses Metal buffers across forward passes. Call `reset()` between passes
/// to recycle all temporary buffers. Buffers are bucketed by power-of-2 size.
public final class BufferPool: @unchecked Sendable {
    /// Buffers currently in use this pass (kept alive so Metal doesn't dealloc).
    private var active: [MTLBuffer] = []
    /// Free buffers by log2(size) bucket.
    private var free: [Int: [MTLBuffer]] = [:]

    /// Get a buffer of at least `length` bytes. May be larger (power-of-2 rounded).
    public func get(length: Int) -> MTLBuffer {
        let size = max(length, 256)
        if var list = free[size], !list.isEmpty {
            let buf = list.removeLast()
            free[size] = list
            active.append(buf)
            return buf
        }
        let buf = MetalContext.shared.device.makeBuffer(length: size, options: .storageModeShared)!
        active.append(buf)
        return buf
    }

    /// Recycle all active buffers back to free lists.
    public func reset() {
        for buf in active {
            free[buf.length, default: []].append(buf)
        }
        active.removeAll(keepingCapacity: true)
    }

    /// Recycle all active buffers EXCEPT the given ones.
    public func reset(keeping: [MTLBuffer]) {
        let keepSet = Set(keeping.map { ObjectIdentifier($0) })
        for buf in active {
            if keepSet.contains(ObjectIdentifier(buf)) { continue }
            free[buf.length, default: []].append(buf)
        }
        active.removeAll(keepingCapacity: true)
        active.append(contentsOf: keeping)
    }
}
