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

    /// Run a compute pass. If batching is active, appends to the current command
    /// buffer with a memory barrier. Otherwise creates+commits its own (legacy sync path).
    public func run(_ body: (MTLComputeCommandEncoder) -> Void) {
        if batching {
            let enc = encoder()
            body(enc)
            return
        }
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        body(enc)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Batched dispatch (multiple ops in one command buffer)

    private var _cmdBuf: MTLCommandBuffer?
    private var _encoder: MTLComputeCommandEncoder?

    /// True when ops should use the batched encoder instead of individual run() calls.
    public private(set) var batching: Bool = false

    /// Begin batching: subsequent calls to encoder() share one command buffer.
    public func beginBatch() {
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        _cmdBuf = cmdBuf
        _encoder = enc
        batching = true
    }

    /// Get the current batched encoder. Inserts a memory barrier between dispatches.
    /// Only valid between beginBatch() and endBatch().
    public func encoder() -> MTLComputeCommandEncoder {
        let enc = _encoder!
        enc.memoryBarrier(scope: .buffers)
        return enc
    }

    /// End batching: commit and wait for all queued work.
    public func endBatch() {
        guard let enc = _encoder, let cmdBuf = _cmdBuf else { return }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        _encoder = nil
        _cmdBuf = nil
        batching = false
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

    /// Purge free lists to actually release GPU memory.
    public func purge() {
        free.removeAll()
    }

    /// Nuclear option: drop ALL buffers (active + free). Use between pipeline stages.
    /// Only buffers in the `keeping` list survive.
    public func releaseAll(keeping: [MTLBuffer] = []) {
        let keepSet = Set(keeping.map { ObjectIdentifier($0) })
        active.removeAll { !keepSet.contains(ObjectIdentifier($0)) }
        free.removeAll()
    }
}
