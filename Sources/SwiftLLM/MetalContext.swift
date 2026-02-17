// Metal device + command queue singleton.

import Metal
import Darwin.Mach

/// Track peak process memory (phys_footprint — what Activity Monitor shows).
public final class PeakMemoryTracker: @unchecked Sendable {
    nonisolated(unsafe) public static let shared = PeakMemoryTracker()
    private var _peak: UInt64 = 0
    private let lock = os_unfair_lock_t.allocate(capacity: 1)

    private init() { lock.initialize(to: os_unfair_lock()) }

    /// Current process physical footprint in bytes.
    public var current: UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    /// Sample current and update peak if higher. Returns (current, peak) in bytes.
    @discardableResult
    public func sample() -> (current: UInt64, peak: UInt64) {
        let c = current
        os_unfair_lock_lock(lock)
        if c > _peak { _peak = c }
        let p = _peak
        os_unfair_lock_unlock(lock)
        return (c, p)
    }

    public var peak: UInt64 {
        os_unfair_lock_lock(lock)
        let p = _peak
        os_unfair_lock_unlock(lock)
        return p
    }

    public func reset() {
        os_unfair_lock_lock(lock)
        _peak = 0
        os_unfair_lock_unlock(lock)
    }
}

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
    /// Free buffers by exact size.
    private var free: [Int: [MTLBuffer]] = [:]

    /// Get a buffer of at least `length` bytes.
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

    /// Trim free lists: keep at most 1 buffer per size to cap memory while allowing reuse.
    public func trimFreeList() {
        for (size, var list) in free {
            if list.count > 1 {
                free[size] = [list.removeLast()]
            }
        }
    }

    /// Nuclear option: drop ALL buffers (active + free). Use between pipeline stages.
    /// Only buffers in the `keeping` list survive.
    public func releaseAll(keeping: [MTLBuffer] = []) {
        let keepSet = Set(keeping.map { ObjectIdentifier($0) })
        active.removeAll { !keepSet.contains(ObjectIdentifier($0)) }
        free.removeAll()
    }

    /// Debug: total bytes in active + free lists.
    public var stats: (active: Int, free: Int, activeCount: Int, freeCount: Int) {
        let freeBytes = free.values.reduce(0) { $0 + $1.reduce(0) { $0 + $1.length } }
        let freeCount = free.values.reduce(0) { $0 + $1.count }
        let activeBytes = active.reduce(0) { $0 + $1.length }
        return (activeBytes, freeBytes, active.count, freeCount)
    }
}
