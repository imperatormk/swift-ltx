// Element-wise Metal kernels for transformer ops.
// These are simple kernels compiled at runtime via makeLibrary(source:).
// The heavy stuff (GEMM, attention) comes from FlashAttention package.

import Metal

/// Compiled kernel cache.
public final class KernelCache: @unchecked Sendable {
    nonisolated(unsafe) public static let shared = KernelCache()

    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let library: MTLLibrary

    private init() {
        let ctx = MetalContext.shared
        do {
            self.library = try ctx.device.makeLibrary(source: KernelCache.metalSource, options: nil)
        } catch {
            fatalError("Metal shader compilation failed: \(error)")
        }
    }

    public func pipeline(_ name: String) -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        let fn = library.makeFunction(name: name)!
        let p = try! MetalContext.shared.device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    // RMS Norm: out[i] = (x[i] / rms) * weight[i]
    // Parallel reduction across threads in threadgroup.
    // Grid: (rows) threadgroups, threadgroupSize threads each.
    // Input/output f16, weight f32, accumulator f32.
    kernel void rms_norm(
        device const half* x [[buffer(0)]],
        device const float* weight [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant uint& dim [[buffer(3)]],
        constant float& eps [[buffer(4)]],
        uint tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]])
    {
        uint row = tg_id;
        uint off = row * dim;

        // Parallel sum of squares
        float sum_sq = 0;
        for (uint i = tid_in_tg; i < dim; i += tg_size) {
            float v = float(x[off + i]);
            sum_sq += v * v;
        }

        // SIMD reduction
        sum_sq += simd_shuffle_xor(sum_sq, 1);
        sum_sq += simd_shuffle_xor(sum_sq, 2);
        sum_sq += simd_shuffle_xor(sum_sq, 4);
        sum_sq += simd_shuffle_xor(sum_sq, 8);
        sum_sq += simd_shuffle_xor(sum_sq, 16);

        threadgroup float shared[8];
        uint simd_lane = tid_in_tg % 32;
        uint simd_id = tid_in_tg / 32;
        if (simd_lane == 0) shared[simd_id] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid_in_tg == 0) {
            float total_sq = 0;
            uint num_simds = (tg_size + 31) / 32;
            for (uint i = 0; i < num_simds; i++) total_sq += shared[i];
            shared[0] = rsqrt(total_sq / float(dim) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rms = shared[0];

        // Apply norm + weight in parallel
        for (uint i = tid_in_tg; i < dim; i += tg_size) {
            out[off + i] = half(float(x[off + i]) * rms * weight[i]);
        }
    }

    // Fused SiLU + element-wise multiply: out = silu(a) * b
    // Saves a kernel launch and memory round-trip.
    kernel void silu_mul(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device half* out [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        float v = float(a[tid]);
        out[tid] = half((v / (1.0f + exp(-v))) * float(b[tid]));
    }

    // SiLU (Swish): out = x * sigmoid(x) = x / (1 + exp(-x))
    kernel void silu(
        device const half* x [[buffer(0)]],
        device half* out [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        float v = float(x[tid]);
        out[tid] = half(v / (1.0f + exp(-v)));
    }

    // Element-wise multiply: out = a * b
    kernel void mul(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device half* out [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        out[tid] = a[tid] * b[tid];
    }

    // Fused residual add: out = a + b (same as add, kept for clarity)
    kernel void add(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device half* out [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        out[tid] = a[tid] + b[tid];
    }

    // Transpose [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim]
    // Grid: (headDim, seqLen * nHeads)
    kernel void transpose_sh(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& seqLen [[buffer(2)]],
        constant uint& nHeads [[buffer(3)]],
        constant uint& headDim [[buffer(4)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint s = idx / nHeads;
        uint h = idx % nHeads;
        dst[h * seqLen * headDim + s * headDim + d] = src[s * nHeads * headDim + h * headDim + d];
    }

    // Transpose [nHeads, seqLen, headDim] → [seqLen, nHeads, headDim]
    kernel void transpose_hs(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& seqLen [[buffer(2)]],
        constant uint& nHeads [[buffer(3)]],
        constant uint& headDim [[buffer(4)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint h = idx / seqLen;
        uint s = idx % seqLen;
        dst[s * nHeads * headDim + h * headDim + d] = src[h * seqLen * headDim + s * headDim + d];
    }

    // RoPE (Rotary Position Embedding) with precomputed frequencies.
    // Uses half-dim split pairing: pair i rotates (x[i], x[i + D/2]).
    // Input/output f16, freqs f32, trig computed in f32.
    kernel void rope(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& headDim [[buffer(2)]],
        constant uint& startPos [[buffer(3)]],
        device const float* freqs [[buffer(4)]],
        constant uint& seqLen [[buffer(5)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint pair = tid.x;
        uint hs = tid.y;
        uint seq_offset = hs % seqLen;
        uint position = startPos + seq_offset;

        uint halfDim = headDim / 2;
        uint idx0 = hs * headDim + pair;
        uint idx1 = idx0 + halfDim;

        float freq = freqs[pair];
        float angle = float(position) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float x0 = float(src[idx0]);
        float x1 = float(src[idx1]);
        dst[idx0] = half(x0 * cos_a - x1 * sin_a);
        dst[idx1] = half(x0 * sin_a + x1 * cos_a);
    }

    // Fused transpose [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim] + RoPE.
    // Reads from [s * nHeads * headDim + h * headDim + d], writes to [h * seqLen * headDim + s * headDim + d]
    // with rotary embedding applied.
    // Grid: (headDim/2, seqLen * nHeads)
    kernel void transpose_sh_rope(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& headDim [[buffer(2)]],
        constant uint& seqLen [[buffer(3)]],
        constant uint& nHeads [[buffer(4)]],
        constant uint& startPos [[buffer(5)]],
        device const float* freqs [[buffer(6)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint pair = tid.x;
        uint idx = tid.y;
        uint s = idx / nHeads;
        uint h = idx % nHeads;

        uint halfDim = headDim / 2;
        uint position = startPos + s;

        // Source indices: [seqLen, nHeads, headDim] layout
        uint src_base = s * nHeads * headDim + h * headDim;
        uint src0 = src_base + pair;
        uint src1 = src_base + pair + halfDim;

        // Dest indices: [nHeads, seqLen, headDim] layout
        uint dst_base = h * seqLen * headDim + s * headDim;
        uint dst0 = dst_base + pair;
        uint dst1 = dst_base + pair + halfDim;

        float freq = freqs[pair];
        float angle = float(position) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float x0 = float(src[src0]);
        float x1 = float(src[src1]);
        dst[dst0] = half(x0 * cos_a - x1 * sin_a);
        dst[dst1] = half(x0 * sin_a + x1 * cos_a);
    }

    // Fused residual add + RMS norm: out = rms_norm(a + b, weight)
    // Saves a kernel launch and one memory round-trip for the intermediate sum.
    // Grid: (rows) threadgroups, threadgroupSize threads each.
    kernel void residual_rms_norm(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device const float* weight [[buffer(2)]],
        device half* residual_out [[buffer(3)]],
        device half* norm_out [[buffer(4)]],
        constant uint& dim [[buffer(5)]],
        constant float& eps [[buffer(6)]],
        uint tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]])
    {
        uint row = tg_id;
        uint off = row * dim;

        // Pass 1: compute a+b and sum of squares
        float sum_sq = 0;
        for (uint i = tid_in_tg; i < dim; i += tg_size) {
            float v = float(a[off + i]) + float(b[off + i]);
            residual_out[off + i] = half(v);
            sum_sq += v * v;
        }

        // SIMD reduction
        sum_sq += simd_shuffle_xor(sum_sq, 1);
        sum_sq += simd_shuffle_xor(sum_sq, 2);
        sum_sq += simd_shuffle_xor(sum_sq, 4);
        sum_sq += simd_shuffle_xor(sum_sq, 8);
        sum_sq += simd_shuffle_xor(sum_sq, 16);

        threadgroup float shared[8];
        uint simd_lane = tid_in_tg % 32;
        uint simd_id = tid_in_tg / 32;
        if (simd_lane == 0) shared[simd_id] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid_in_tg == 0) {
            float total_sq = 0;
            uint num_simds = (tg_size + 31) / 32;
            for (uint i = 0; i < num_simds; i++) total_sq += shared[i];
            shared[0] = rsqrt(total_sq / float(dim) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rms = shared[0];

        // Pass 2: apply norm + weight (read from residual_out to avoid recomputing a+b)
        for (uint i = tid_in_tg; i < dim; i += tg_size) {
            norm_out[off + i] = half(float(residual_out[off + i]) * rms * weight[i]);
        }
    }

    // Copy a contiguous slice from src to dst (uint4 = 16 bytes per thread).
    kernel void copy_slice(
        device const uint4* src [[buffer(0)]],
        device uint4* dst [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        dst[tid] = src[tid];
    }

    // GPU zero-fill (uint4 = 16 bytes per thread for bandwidth).
    kernel void fill_zero(
        device uint4* dst [[buffer(0)]],
        uint tid [[thread_position_in_grid]])
    {
        dst[tid] = uint4(0);
    }

    // KV cache append: copy [numKVHeads, seqLen, headDim] into strided [numKVHeads, maxSeqLen, headDim]
    // Grid: (headDim, numKVHeads * seqLen) threads
    kernel void kv_append(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& headDim [[buffer(2)]],
        constant uint& seqLen [[buffer(3)]],
        constant uint& maxSeqLen [[buffer(4)]],
        constant uint& startPos [[buffer(5)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint h = idx / seqLen;
        uint s = idx % seqLen;
        uint src_off = h * seqLen * headDim + s * headDim + d;
        uint dst_off = h * maxSeqLen * headDim + (startPos + s) * headDim + d;
        dst[dst_off] = src[src_off];
    }

    // Fused transpose + KV cache append:
    // Reads from [seqLen, nHeads, headDim], writes into strided [nHeads, maxSeqLen, headDim].
    // Eliminates separate transpose_sh dispatch for V before kv_append.
    // Grid: (headDim, nHeads * seqLen)
    kernel void kv_append_transposed(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& headDim [[buffer(2)]],
        constant uint& seqLen [[buffer(3)]],
        constant uint& maxSeqLen [[buffer(4)]],
        constant uint& startPos [[buffer(5)]],
        constant uint& nHeads [[buffer(6)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint h = idx / seqLen;
        uint s = idx % seqLen;
        // src layout: [seqLen, nHeads, headDim]
        uint src_off = s * nHeads * headDim + h * headDim + d;
        // dst layout: [nHeads, maxSeqLen, headDim]
        uint dst_off = h * maxSeqLen * headDim + (startPos + s) * headDim + d;
        dst[dst_off] = src[src_off];
    }

    // Softmax over last dimension (for logits → probs).
    kernel void softmax_1d(
        device float* x [[buffer(0)]],
        constant uint& n [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid != 0) return;

        float max_val = -INFINITY;
        for (uint i = 0; i < n; i++) max_val = max(max_val, x[i]);

        float sum = 0;
        for (uint i = 0; i < n; i++) {
            x[i] = exp(x[i] - max_val);
            sum += x[i];
        }
        for (uint i = 0; i < n; i++) x[i] /= sum;
    }

    // Batch embedding lookup: out[t * dim + i] = table[token_ids[t] * dim + i]
    // Table stays f32 (weight data), output f16.
    kernel void embedding_batch(
        device const float* table [[buffer(0)]],
        device half* out [[buffer(1)]],
        device const uint* token_ids [[buffer(2)]],
        constant uint& dim [[buffer(3)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint i = tid.x;
        uint t = tid.y;
        out[t * dim + i] = half(table[token_ids[t] * dim + i]);
    }

    // Batch quantized embedding lookup — output f16.
    kernel void embedding_q4_batch(
        device const uint* weight [[buffer(0)]],
        device const half* scales [[buffer(1)]],
        device const half* biases [[buffer(2)]],
        device half* out [[buffer(3)]],
        device const uint* token_ids [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant uint& group_size [[buffer(6)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint k_idx = tid.x;
        uint t = tid.y;
        uint token_id = token_ids[t];
        uint packed_k = K / 8;
        uint groups_per_row = K / group_size;

        uint pack_idx = k_idx / 8;
        uint sub_idx = k_idx % 8;
        uint group_idx = k_idx / group_size;

        uint packed_val = weight[token_id * packed_k + pack_idx];
        uint nibble = (packed_val >> (sub_idx * 4)) & 0xF;

        float scale = float(scales[token_id * groups_per_row + group_idx]);
        float bias = float(biases[token_id * groups_per_row + group_idx]);

        out[t * K + k_idx] = half(float(nibble) * scale + bias);
    }

    // Embedding lookup — table f32, output f16.
    kernel void embedding(
        device const float* table [[buffer(0)]],
        device half* out [[buffer(1)]],
        constant uint& token_id [[buffer(2)]],
        constant uint& dim [[buffer(3)]],
        uint tid [[thread_position_in_grid]])
    {
        out[tid] = half(table[token_id * dim + tid]);
    }

    // Quantized embedding lookup (4-bit) — output f16.
    kernel void embedding_q4(
        device const uint* weight [[buffer(0)]],
        device const half* scales [[buffer(1)]],
        device const half* biases [[buffer(2)]],
        device half* out [[buffer(3)]],
        constant uint& token_id [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant uint& group_size [[buffer(6)]],
        uint tid [[thread_position_in_grid]])
    {
        uint packed_k = K / 8;
        uint groups_per_row = K / group_size;

        uint pack_idx = tid / 8;
        uint sub_idx = tid % 8;
        uint group_idx = tid / group_size;

        uint packed_val = weight[token_id * packed_k + pack_idx];
        uint nibble = (packed_val >> (sub_idx * 4)) & 0xF;

        float scale = float(scales[token_id * groups_per_row + group_idx]);
        float bias = float(biases[token_id * groups_per_row + group_idx]);

        out[tid] = half(float(nibble) * scale + bias);
    }

    // Quantized matvec: out = x @ W^T where W is 4-bit quantized.
    // Inspired by MLX/llama.cpp: 2 simdgroups × 4 rows = 8 output rows per threadgroup.
    // Each thread processes 2 packed uint32s (16 weights) per K-iteration.
    // Uses MLX's delayed-division trick: pre-scale x, mask without shifting.
    // Input x is f16, accumulator f32, output f16.
    //
    // Grid: ceil(M * N / 8) threadgroups, 64 threads each.
    // params: [K, group_size, N, M]
    kernel void matmul_q4(
        device const half* x [[buffer(0)]],
        device const uint* weight [[buffer(1)]],
        device const half* scales [[buffer(2)]],
        device const half* biases [[buffer(3)]],
        device half* out [[buffer(4)]],
        constant uint* params [[buffer(5)]],
        uint tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]])
    {
        constexpr uint NUM_SIMDGROUPS = 2;
        constexpr uint ROWS_PER_SG = 4;
        constexpr uint ROWS_PER_TG = NUM_SIMDGROUPS * ROWS_PER_SG;  // 8

        uint K = params[0];
        uint group_size = params[1];
        uint N = params[2];
        uint M = params[3];

        uint packed_k = K / 8;
        uint groups_per_row = K / group_size;
        uint packs_per_group = group_size / 8;

        uint out_base = tg_id * ROWS_PER_TG;
        uint total_out = M * N;
        uint sg_row_base = out_base + simd_gid * ROWS_PER_SG;

        // Per-row: dot product accumulator and bias*sum_x accumulator
        float sums[ROWS_PER_SG] = {0, 0, 0, 0};
        float bias_sums[ROWS_PER_SG] = {0, 0, 0, 0};

        // Precompute (m, n) for each row in this simdgroup
        uint row_n[ROWS_PER_SG];
        uint row_m[ROWS_PER_SG];
        uint valid_rows = 0;
        for (uint ri = 0; ri < ROWS_PER_SG; ri++) {
            uint out_idx = sg_row_base + ri;
            if (out_idx >= total_out) break;
            row_n[ri] = out_idx % N;
            row_m[ri] = out_idx / N;
            valid_rows = ri + 1;
        }

        // 4 packs per lane per iteration = 32 weights, stride 128 packs across 32 lanes
        for (uint p_base = simd_lane * 4; p_base < packed_k; p_base += 128) {
            uint k_off = p_base * 8;

            // Load 32 x values for m=0 (decode common case)
            device const half* x_ptr = x + k_off;
            half4 xv[8];
            xv[0] = *((device const half4*)(x_ptr));
            xv[1] = *((device const half4*)(x_ptr + 4));
            xv[2] = *((device const half4*)(x_ptr + 8));
            xv[3] = *((device const half4*)(x_ptr + 12));
            xv[4] = *((device const half4*)(x_ptr + 16));
            xv[5] = *((device const half4*)(x_ptr + 20));
            xv[6] = *((device const half4*)(x_ptr + 24));
            xv[7] = *((device const half4*)(x_ptr + 28));

            // sum_x per pack for bias factoring
            float sum_x0 = float(xv[0].x) + float(xv[0].y) + float(xv[0].z) + float(xv[0].w)
                         + float(xv[1].x) + float(xv[1].y) + float(xv[1].z) + float(xv[1].w);
            float sum_x1 = float(xv[2].x) + float(xv[2].y) + float(xv[2].z) + float(xv[2].w)
                         + float(xv[3].x) + float(xv[3].y) + float(xv[3].z) + float(xv[3].w);
            float sum_x2 = float(xv[4].x) + float(xv[4].y) + float(xv[4].z) + float(xv[4].w)
                         + float(xv[5].x) + float(xv[5].y) + float(xv[5].z) + float(xv[5].w);
            float sum_x3 = float(xv[6].x) + float(xv[6].y) + float(xv[6].z) + float(xv[6].w)
                         + float(xv[7].x) + float(xv[7].y) + float(xv[7].z) + float(xv[7].w);

            for (uint ri = 0; ri < valid_rows; ri++) {
                uint n = row_n[ri];
                uint m = row_m[ri];

                if (m > 0) {
                    device const half* mx_ptr = x + m * K + k_off;
                    xv[0] = *((device const half4*)(mx_ptr));
                    xv[1] = *((device const half4*)(mx_ptr + 4));
                    xv[2] = *((device const half4*)(mx_ptr + 8));
                    xv[3] = *((device const half4*)(mx_ptr + 12));
                    xv[4] = *((device const half4*)(mx_ptr + 16));
                    xv[5] = *((device const half4*)(mx_ptr + 20));
                    xv[6] = *((device const half4*)(mx_ptr + 24));
                    xv[7] = *((device const half4*)(mx_ptr + 28));
                    sum_x0 = float(xv[0].x) + float(xv[0].y) + float(xv[0].z) + float(xv[0].w)
                           + float(xv[1].x) + float(xv[1].y) + float(xv[1].z) + float(xv[1].w);
                    sum_x1 = float(xv[2].x) + float(xv[2].y) + float(xv[2].z) + float(xv[2].w)
                           + float(xv[3].x) + float(xv[3].y) + float(xv[3].z) + float(xv[3].w);
                    sum_x2 = float(xv[4].x) + float(xv[4].y) + float(xv[4].z) + float(xv[4].w)
                           + float(xv[5].x) + float(xv[5].y) + float(xv[5].z) + float(xv[5].w);
                    sum_x3 = float(xv[6].x) + float(xv[6].y) + float(xv[6].z) + float(xv[6].w)
                           + float(xv[7].x) + float(xv[7].y) + float(xv[7].z) + float(xv[7].w);
                }

                device const uint* w_row = weight + n * packed_k;
                uint scale_base = n * groups_per_row;
                uint bias_base = n * groups_per_row;

                // Pack 0
                uint g0 = p_base / packs_per_group;
                float s0 = float(scales[scale_base + g0]);
                uint pv0 = w_row[p_base];
                sums[ri] += float((pv0      ) & 0xF) * s0 * float(xv[0].x);
                sums[ri] += float((pv0 >>  4) & 0xF) * s0 * float(xv[0].y);
                sums[ri] += float((pv0 >>  8) & 0xF) * s0 * float(xv[0].z);
                sums[ri] += float((pv0 >> 12) & 0xF) * s0 * float(xv[0].w);
                sums[ri] += float((pv0 >> 16) & 0xF) * s0 * float(xv[1].x);
                sums[ri] += float((pv0 >> 20) & 0xF) * s0 * float(xv[1].y);
                sums[ri] += float((pv0 >> 24) & 0xF) * s0 * float(xv[1].z);
                sums[ri] += float((pv0 >> 28)      ) * s0 * float(xv[1].w);

                // Pack 1
                uint g1 = (p_base + 1) / packs_per_group;
                float s1 = float(scales[scale_base + g1]);
                uint pv1 = w_row[p_base + 1];
                sums[ri] += float((pv1      ) & 0xF) * s1 * float(xv[2].x);
                sums[ri] += float((pv1 >>  4) & 0xF) * s1 * float(xv[2].y);
                sums[ri] += float((pv1 >>  8) & 0xF) * s1 * float(xv[2].z);
                sums[ri] += float((pv1 >> 12) & 0xF) * s1 * float(xv[2].w);
                sums[ri] += float((pv1 >> 16) & 0xF) * s1 * float(xv[3].x);
                sums[ri] += float((pv1 >> 20) & 0xF) * s1 * float(xv[3].y);
                sums[ri] += float((pv1 >> 24) & 0xF) * s1 * float(xv[3].z);
                sums[ri] += float((pv1 >> 28)      ) * s1 * float(xv[3].w);

                // Pack 2
                uint g2 = (p_base + 2) / packs_per_group;
                float s2 = float(scales[scale_base + g2]);
                uint pv2 = w_row[p_base + 2];
                sums[ri] += float((pv2      ) & 0xF) * s2 * float(xv[4].x);
                sums[ri] += float((pv2 >>  4) & 0xF) * s2 * float(xv[4].y);
                sums[ri] += float((pv2 >>  8) & 0xF) * s2 * float(xv[4].z);
                sums[ri] += float((pv2 >> 12) & 0xF) * s2 * float(xv[4].w);
                sums[ri] += float((pv2 >> 16) & 0xF) * s2 * float(xv[5].x);
                sums[ri] += float((pv2 >> 20) & 0xF) * s2 * float(xv[5].y);
                sums[ri] += float((pv2 >> 24) & 0xF) * s2 * float(xv[5].z);
                sums[ri] += float((pv2 >> 28)      ) * s2 * float(xv[5].w);

                // Pack 3
                uint g3 = (p_base + 3) / packs_per_group;
                float s3 = float(scales[scale_base + g3]);
                uint pv3 = w_row[p_base + 3];
                sums[ri] += float((pv3      ) & 0xF) * s3 * float(xv[6].x);
                sums[ri] += float((pv3 >>  4) & 0xF) * s3 * float(xv[6].y);
                sums[ri] += float((pv3 >>  8) & 0xF) * s3 * float(xv[6].z);
                sums[ri] += float((pv3 >> 12) & 0xF) * s3 * float(xv[6].w);
                sums[ri] += float((pv3 >> 16) & 0xF) * s3 * float(xv[7].x);
                sums[ri] += float((pv3 >> 20) & 0xF) * s3 * float(xv[7].y);
                sums[ri] += float((pv3 >> 24) & 0xF) * s3 * float(xv[7].z);
                sums[ri] += float((pv3 >> 28)      ) * s3 * float(xv[7].w);

                // Bias: once per pack instead of 8 times
                float b0 = float(biases[bias_base + g0]);
                float b1 = float(biases[bias_base + g1]);
                float b2 = float(biases[bias_base + g2]);
                float b3 = float(biases[bias_base + g3]);
                bias_sums[ri] += b0 * sum_x0 + b1 * sum_x1 + b2 * sum_x2 + b3 * sum_x3;
            }
        }

        // SIMD reduction
        for (uint ri = 0; ri < ROWS_PER_SG; ri++) {
            sums[ri] += bias_sums[ri];
            sums[ri] += simd_shuffle_xor(sums[ri], 1);
            sums[ri] += simd_shuffle_xor(sums[ri], 2);
            sums[ri] += simd_shuffle_xor(sums[ri], 4);
            sums[ri] += simd_shuffle_xor(sums[ri], 8);
            sums[ri] += simd_shuffle_xor(sums[ri], 16);
        }

        if (simd_lane == 0) {
            for (uint ri = 0; ri < valid_rows; ri++) {
                uint out_idx = sg_row_base + ri;
                out[out_idx] = half(sums[ri]);
            }
        }
    }

    // Naive GPU attention with causal mask — parallelized across D dimension.
    // Q=[nHeads, R, D], K/V=[nKVHeads, maxSeqLen, D] (strided cache, first C rows valid)
    // Grid: (nHeads, R) threadgroups, D threads each.
    // params: [nHeads, nKVHeads, R, C, D, startPos, maxSeqLen]
    // Input Q/K/V f16, accumulator f32, output O f16.
    kernel void naive_attention(
        device const half* Q [[buffer(0)]],
        device const half* K [[buffer(1)]],
        device const half* V [[buffer(2)]],
        device half* O [[buffer(3)]],
        constant uint* params [[buffer(4)]],
        uint2 tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint2 tg_size2 [[threads_per_threadgroup]])
    {
        uint tg_size = tg_size2.x;
        uint h = tg_id.x;
        uint r = tg_id.y;
        uint nHeads = params[0];
        uint nKVHeads = params[1];
        uint R = params[2];
        uint C = params[3];
        uint D = params[4];
        uint startPos = params[5];
        uint maxSeqLen = params[6];
        uint kvRepeat = nHeads / nKVHeads;
        uint kvh = h / kvRepeat;

        uint causal_limit = min(startPos + r + 1, C);
        float scale = rsqrt(float(D));

        uint q_base = h * R * D + r * D;
        uint kv_base = kvh * maxSeqLen * D;

        // Shared memory for reductions (must be at function scope)
        threadgroup float tg_reduce[8];
        threadgroup float tg_broadcast;

        // Each thread handles a subset of D dimensions
        float q_local[4];
        uint d_per_thread = (D + tg_size - 1) / tg_size;
        uint d_start = tid_in_tg * d_per_thread;
        uint d_end = min(d_start + d_per_thread, D);
        uint d_count = (d_start < D) ? (d_end - d_start) : 0;
        for (uint i = 0; i < d_count && i < 4; i++) {
            q_local[i] = float(Q[q_base + d_start + i]);
        }

        uint simd_lane = tid_in_tg % 32;
        uint simd_id = tid_in_tg / 32;

        // Helper: reduce partial_dot across all threads, broadcast result
        // After this, `dot` is the same value in ALL threads.
        #define REDUCE_DOT(partial_dot, result) \
            partial_dot += simd_shuffle_xor(partial_dot, 1); \
            partial_dot += simd_shuffle_xor(partial_dot, 2); \
            partial_dot += simd_shuffle_xor(partial_dot, 4); \
            partial_dot += simd_shuffle_xor(partial_dot, 8); \
            partial_dot += simd_shuffle_xor(partial_dot, 16); \
            if (simd_lane == 0) tg_reduce[simd_id] = partial_dot; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (tid_in_tg == 0) { \
                float _s = 0; \
                uint _n = (tg_size + 31) / 32; \
                for (uint _i = 0; _i < _n; _i++) _s += tg_reduce[_i]; \
                tg_broadcast = _s; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            float result = tg_broadcast;

        // Pass 1: find max score
        float max_s = -INFINITY;
        for (uint s = 0; s < causal_limit; s++) {
            float partial_dot = 0;
            for (uint i = 0; i < d_count && i < 4; i++) {
                partial_dot += q_local[i] * float(K[kv_base + s * D + d_start + i]);
            }
            REDUCE_DOT(partial_dot, dot)
            max_s = max(max_s, dot * scale);
        }

        // Pass 2: softmax + weighted V accumulation
        float sum_e = 0;
        float o_acc[4] = {0, 0, 0, 0};
        for (uint s = 0; s < causal_limit; s++) {
            float partial_dot = 0;
            for (uint i = 0; i < d_count && i < 4; i++) {
                partial_dot += q_local[i] * float(K[kv_base + s * D + d_start + i]);
            }
            REDUCE_DOT(partial_dot, dot2)
            float p = exp(dot2 * scale - max_s);
            sum_e += p;
            for (uint i = 0; i < d_count && i < 4; i++) {
                o_acc[i] += p * float(V[kv_base + s * D + d_start + i]);
            }
        }

        #undef REDUCE_DOT

        // Write output
        float inv_sum = 1.0f / sum_e;
        uint o_base = h * R * D + r * D;
        for (uint i = 0; i < d_count && i < 4; i++) {
            O[o_base + d_start + i] = half(o_acc[i] * inv_sum);
        }
    }
    """
}
