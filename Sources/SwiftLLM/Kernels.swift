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
    #include <metal_simdgroup_matrix>
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

    // ========== SIMD GEMM (no per-shape compilation) ==========
    // C[M,N] = A[M,K] × B^T[N,K] using simdgroup_matrix hardware.
    // Each threadgroup: 4 simdgroups, computes 32×32 output tile.
    // Each simdgroup handles 8 M rows × 32 N cols = 4 accumulator tiles.
    // Grid: (ceil(N/32), ceil(M/32)), threadgroup: (32, 4) = 128 threads

    // Uses threadgroup memory for bounds-safe loading, then simdgroup_load from smem.
    // smem layout: [32][8] for A tile + [8][32] for B^T tile = 512 + 512 = 1024 bytes per K step
    // But we keep smem persistent: [32][8] A + [32][8] B = 1024 halfs = 2KB total

    kernel void simple_gemm_f16(
        device const half* A [[buffer(0)]],
        device const half* B [[buffer(1)]],
        device half* C [[buffer(2)]],
        constant uint& M [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint simd_id [[simdgroup_index_in_threadgroup]],
        uint lane_id [[thread_index_in_simdgroup]],
        threadgroup half* smem [[threadgroup(0)]])
    {
        uint tile_m = tgid.y * 32;
        uint tile_n = tgid.x * 32;
        uint sg_m = tile_m + simd_id * 8;
        uint flat_tid = simd_id * 32 + lane_id;  // 0..127

        // smem layout: A_smem[32][8] then BT_smem[8][32]
        // A_smem: rows = M local, cols = K chunk (8)
        // BT_smem: rows = K chunk (8), cols = N local (32) — already transposed
        threadgroup half* A_smem = smem;           // 32*8 = 256 halfs
        threadgroup half* BT_smem = smem + 32 * 8; // 8*32 = 256 halfs

        simdgroup_matrix<float, 8, 8> acc0(0), acc1(0), acc2(0), acc3(0);

        for (uint k_start = 0; k_start < K; k_start += 8) {
            // Cooperatively load A[tile_m:tile_m+32, k_start:k_start+8] into A_smem[32][8]
            // 128 threads, 256 elements -> 2 elements per thread
            for (uint i = flat_tid; i < 32 * 8; i += 128) {
                uint r = i / 8;
                uint c = i % 8;
                uint mr = tile_m + r;
                uint kc = k_start + c;
                A_smem[r * 8 + c] = (mr < M && kc < K) ? A[mr * K + kc] : half(0);
            }

            // Load B^T[k_start:k_start+8, tile_n:tile_n+32] into BT_smem[8][32]
            // B is [N,K], B^T[k,n] = B[n,k]
            for (uint i = flat_tid; i < 8 * 32; i += 128) {
                uint kr = i / 32;  // k row (0..7)
                uint nc = i % 32;  // n col (0..31)
                uint kc = k_start + kr;
                uint nr = tile_n + nc;
                BT_smem[kr * 32 + nc] = (kc < K && nr < N) ? B[nr * K + kc] : half(0);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each simdgroup loads its 8×8 A tile from smem
            simdgroup_matrix<half, 8, 8> a_tile;
            simdgroup_load(a_tile, A_smem + simd_id * 8 * 8, 8);

            // Load 4 B^T tiles [8][8] from BT_smem
            simdgroup_matrix<half, 8, 8> bt0, bt1, bt2, bt3;
            simdgroup_load(bt0, BT_smem + 0,  32);
            simdgroup_load(bt1, BT_smem + 8,  32);
            simdgroup_load(bt2, BT_smem + 16, 32);
            simdgroup_load(bt3, BT_smem + 24, 32);

            simdgroup_multiply_accumulate(acc0, a_tile, bt0, acc0);
            simdgroup_multiply_accumulate(acc1, a_tile, bt1, acc1);
            simdgroup_multiply_accumulate(acc2, a_tile, bt2, acc2);
            simdgroup_multiply_accumulate(acc3, a_tile, bt3, acc3);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store results: each simdgroup writes 8×32 output (4 × 8×8 tiles)
        // Use float smem for simdgroup_store, then convert to half when writing to C
        threadgroup float* out_f = (threadgroup float*)smem;  // 32*32 floats = 4096 bytes
        simdgroup_store(acc0, out_f + simd_id * 8 * 32 + 0,  32);
        simdgroup_store(acc1, out_f + simd_id * 8 * 32 + 8,  32);
        simdgroup_store(acc2, out_f + simd_id * 8 * 32 + 16, 32);
        simdgroup_store(acc3, out_f + simd_id * 8 * 32 + 24, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Copy from smem to global with bounds check, converting f32→f16
        for (uint i = flat_tid; i < 32 * 32; i += 128) {
            uint r = i / 32;
            uint c = i % 32;
            uint mr = tile_m + r;
            uint nc = tile_n + c;
            if (mr < M && nc < N) {
                C[mr * N + nc] = half(out_f[r * 32 + c]);
            }
        }
    }

    // ========== VAE 3D Decoder Kernels ==========

    // Scatter GEMM output [M_chunk, C_out] (row-major) into channel-first output [B, C_out, D, H, W].
    // Adds bias on first chunk call (addBias=1), accumulates on subsequent (addBias=0 means overwrite, =2 means accumulate).
    // mode: 0 = write (first kernel pos), 1 = accumulate (subsequent kernel pos)
    // params: C_out, D_out, H_out, W_out, m_offset, M_chunk, mode
    // Grid: dispatchThreads(C_out, M_chunk, 1)
    kernel void scatter_conv_output(
        device const half* gemm_out [[buffer(0)]],
        device const half* bias [[buffer(1)]],
        device half* output [[buffer(2)]],
        constant uint* params [[buffer(3)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint C_out = params[0];
        uint D_out = params[1], H_out = params[2], W_out = params[3];
        uint m_offset = params[4];
        uint M_chunk = params[5];
        uint mode = params[6];  // 0 = write with bias, 1 = accumulate

        uint oc = tid.x;
        uint row = tid.y;
        if (oc >= C_out || row >= M_chunk) return;

        uint m = m_offset + row;
        uint HW_out = H_out * W_out;
        uint DHW_out = D_out * HW_out;
        uint b = m / DHW_out;
        uint spatial = m % DHW_out;

        float val = float(gemm_out[row * C_out + oc]);
        uint out_idx = b * C_out * DHW_out + oc * DHW_out + spatial;

        if (mode == 0) {
            output[out_idx] = half(val + float(bias[oc]));
        } else {
            output[out_idx] = half(float(output[out_idx]) + val);
        }
    }

    // Im2col for 3D convolution: gathers all kernel positions into [M_chunk, C_in * kSize] matrix.
    // Each thread handles one element of the output matrix.
    // params: C_in, D_in, H_in, W_in, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW, m_offset, M_chunk
    // Grid: dispatchThreads(C_in * kSize, M_chunk, 1)
    kernel void im2col_3d(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint* params [[buffer(2)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint C_in = params[0];
        uint D_in = params[1], H_in = params[2], W_in = params[3];
        uint D_out = params[4], H_out = params[5], W_out = params[6];
        uint kD = params[7], kH = params[8], kW = params[9];
        uint sD = params[10], sH = params[11], sW = params[12];
        uint pD = params[13], pH = params[14], pW = params[15];
        uint m_offset = params[16];
        uint M_chunk = params[17];

        uint col = tid.x;  // which column: ic * kSize + kernel_pos
        uint row = tid.y;  // which row in chunk

        uint kSize = kD * kH * kW;
        uint total_cols = C_in * kSize;
        if (col >= total_cols || row >= M_chunk) return;

        uint m = m_offset + row;  // global output spatial index
        // Decompose m into (b, od, oh, ow)
        uint HW_out = H_out * W_out;
        uint DHW_out = D_out * HW_out;
        uint b = m / DHW_out;
        uint rem = m % DHW_out;
        uint od = rem / HW_out;
        uint rem2 = rem % HW_out;
        uint oh = rem2 / W_out;
        uint ow = rem2 % W_out;

        // Decompose col into (ic, fd, fh, fw)
        uint ic = col / kSize;
        uint k_rem = col % kSize;
        uint kHW = kH * kW;
        uint fd = k_rem / kHW;
        uint k_rem2 = k_rem % kHW;
        uint fh = k_rem2 / kW;
        uint fw = k_rem2 % kW;

        // Compute input position
        int id = int(od * sD + fd) - int(pD);
        int ih = int(oh * sH + fh) - int(pH);
        int iw = int(ow * sW + fw) - int(pW);

        half val = 0;
        if (id >= 0 && uint(id) < D_in && ih >= 0 && uint(ih) < H_in && iw >= 0 && uint(iw) < W_in) {
            uint HW_in = H_in * W_in;
            val = input[b * C_in * D_in * HW_in + ic * D_in * HW_in + uint(id) * HW_in + uint(ih) * W_in + uint(iw)];
        }

        output[row * total_cols + col] = val;
    }

    // 1x1x1 convolution (pointwise) — input-stationary with shared memory.
    // Each threadgroup: one spatial position, all C_out channels.
    // Load C_in input values into smem, each thread does one dot product.
    // Grid: dispatchThreadgroups(D*H*W, B, 1), threadsPerThreadgroup(min(C_out, 256), 1, 1)
    kernel void conv3d_1x1x1(
        device const half* input [[buffer(0)]],
        device const half* weight [[buffer(1)]],
        device const half* bias [[buffer(2)]],
        device half* output [[buffer(3)]],
        constant uint* params [[buffer(4)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint2 tg_size2 [[threads_per_threadgroup]],
        threadgroup half* smem [[threadgroup(0)]])
    {
        uint C_in = params[0];
        uint spatial_size = params[1];
        uint C_out = params[2];
        uint TILE_C = params[3];

        uint tg_size = tg_size2.x;
        uint s = tgid.x;
        uint b = tgid.y;

        uint in_base = b * C_in * spatial_size + s;
        uint ocs_per_thread = (C_out + tg_size - 1) / tg_size;

        float acc[4] = {0, 0, 0, 0};

        for (uint c_start = 0; c_start < C_in; c_start += TILE_C) {
            uint c_end = min(c_start + TILE_C, C_in);
            uint tile_c = c_end - c_start;

            // Cooperatively load tile_c input values
            for (uint i = tid; i < tile_c; i += tg_size) {
                smem[i] = input[in_base + (c_start + i) * spatial_size];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint o = 0; o < ocs_per_thread; o++) {
                uint oc = tid + o * tg_size;
                if (oc >= C_out) break;
                float s2 = 0;
                uint w_base = oc * C_in + c_start;
                for (uint i = 0; i < tile_c; i++) {
                    s2 += float(smem[i]) * float(weight[w_base + i]);
                }
                acc[o] += s2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint out_base = b * C_out * spatial_size + s;
        for (uint o = 0; o < ocs_per_thread; o++) {
            uint oc = tid + o * tg_size;
            if (oc >= C_out) break;
            output[out_base + oc * spatial_size] = half(acc[o] + float(bias[oc]));
        }
    }

    // Pixel norm: x / sqrt(mean(x^2, dim=C) + eps)
    // Input/Output: [B, C, D, H, W] f16
    // Grid: (D*H*W, B)
    kernel void pixel_norm_3d(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint& C [[buffer(2)]],
        constant uint& spatial_size [[buffer(3)]],
        constant float& eps [[buffer(4)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint s = tid.x;
        uint b = tid.y;
        if (s >= spatial_size) return;

        float sum_sq = 0;
        uint base = b * C * spatial_size + s;
        for (uint c = 0; c < C; c++) {
            float v = float(input[base + c * spatial_size]);
            sum_sq += v * v;
        }
        float inv_rms = rsqrt(sum_sq / float(C) + eps);
        for (uint c = 0; c < C; c++) {
            uint idx = base + c * spatial_size;
            output[idx] = half(float(input[idx]) * inv_rms);
        }
    }

    // Group norm 3D: normalize over (C/G, D, H, W) per group, with learned weight+bias.
    // Input: [B, C, D, H, W] f16, Weight/Bias: [C] f32
    // Grid: (numGroups, B) threadgroups, 256 threads each
    kernel void group_norm_3d(
        device const half* input [[buffer(0)]],
        device const float* weight [[buffer(1)]],
        device const float* gn_bias [[buffer(2)]],
        device half* output [[buffer(3)]],
        constant uint& C [[buffer(4)]],
        constant uint& spatial_size [[buffer(5)]],
        constant uint& num_groups [[buffer(6)]],
        constant float& eps [[buffer(7)]],
        uint2 tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint2 tg_size2 [[threads_per_threadgroup]])
    {
        uint tg_size = tg_size2.x;
        uint g = tg_id.x;
        uint b = tg_id.y;
        uint channels_per_group = C / num_groups;
        uint group_size = channels_per_group * spatial_size;

        uint base = b * C * spatial_size + g * channels_per_group * spatial_size;

        // Parallel sum and sum_sq
        float local_sum = 0;
        float local_sum_sq = 0;
        for (uint i = tid_in_tg; i < group_size; i += tg_size) {
            float v = float(input[base + i]);
            local_sum += v;
            local_sum_sq += v * v;
        }

        // SIMD reduction
        local_sum += simd_shuffle_xor(local_sum, 1);
        local_sum += simd_shuffle_xor(local_sum, 2);
        local_sum += simd_shuffle_xor(local_sum, 4);
        local_sum += simd_shuffle_xor(local_sum, 8);
        local_sum += simd_shuffle_xor(local_sum, 16);
        local_sum_sq += simd_shuffle_xor(local_sum_sq, 1);
        local_sum_sq += simd_shuffle_xor(local_sum_sq, 2);
        local_sum_sq += simd_shuffle_xor(local_sum_sq, 4);
        local_sum_sq += simd_shuffle_xor(local_sum_sq, 8);
        local_sum_sq += simd_shuffle_xor(local_sum_sq, 16);

        threadgroup float shared[16];
        uint simd_lane = tid_in_tg % 32;
        uint simd_id = tid_in_tg / 32;
        if (simd_lane == 0) {
            shared[simd_id * 2] = local_sum;
            shared[simd_id * 2 + 1] = local_sum_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid_in_tg == 0) {
            float total_sum = 0, total_sum_sq = 0;
            uint num_simds = (tg_size + 31) / 32;
            for (uint i = 0; i < num_simds; i++) {
                total_sum += shared[i * 2];
                total_sum_sq += shared[i * 2 + 1];
            }
            float mean = total_sum / float(group_size);
            float variance = total_sum_sq / float(group_size) - mean * mean;
            shared[0] = mean;
            shared[1] = rsqrt(variance + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = shared[0];
        float inv_std = shared[1];

        // Apply norm + weight + bias
        for (uint i = tid_in_tg; i < group_size; i += tg_size) {
            uint c_local = i / spatial_size;
            uint c_global = g * channels_per_group + c_local;
            float v = (float(input[base + i]) - mean) * inv_std;
            output[base + i] = half(v * weight[c_global] + gn_bias[c_global]);
        }
    }

    // 3D Pixel shuffle: [B, C*p1*p2*p3, D, H, W] -> [B, C, D*p1, H*p2, W*p3]
    // Grid: total output elements
    kernel void pixel_shuffle_3d(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint* params [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        uint C_out = params[0];
        uint D_out = params[1], H_out = params[2], W_out = params[3];
        uint p1 = params[4], p2 = params[5], p3 = params[6];
        uint B = params[7];
        uint D_in = D_out / p1, H_in = H_out / p2, W_in = W_out / p3;
        uint C_in = C_out * p1 * p2 * p3;
        uint out_spatial = D_out * H_out * W_out;
        uint total = B * C_out * out_spatial;
        if (tid >= total) return;

        uint rem = tid;
        uint b = rem / (C_out * out_spatial); rem %= (C_out * out_spatial);
        uint c = rem / out_spatial; rem %= out_spatial;
        uint od = rem / (H_out * W_out); rem %= (H_out * W_out);
        uint oh = rem / W_out;
        uint ow = rem % W_out;

        uint id = od / p1, r1 = od % p1;
        uint ih = oh / p2, r2 = oh % p2;
        uint iw = ow / p3, r3 = ow % p3;

        uint ic = c * p1 * p2 * p3 + r1 * p2 * p3 + r2 * p3 + r3;
        uint in_idx = b * C_in * D_in * H_in * W_in + ic * D_in * H_in * W_in + id * H_in * W_in + ih * W_in + iw;
        output[tid] = input[in_idx];
    }

    // Scale-shift (AdaLN): out = x * (1 + scale) + shift
    // x: [B, C, D, H, W], scale/shift: [B, C, 1, 1, 1] broadcast
    // Grid: total elements
    kernel void scale_shift_3d(
        device const half* x [[buffer(0)]],
        device const half* scale [[buffer(1)]],
        device const half* shift [[buffer(2)]],
        device half* output [[buffer(3)]],
        constant uint& C [[buffer(4)]],
        constant uint& spatial_size [[buffer(5)]],
        uint tid [[thread_position_in_grid]])
    {
        uint total = tid;
        uint b = total / (C * spatial_size);
        uint c = (total / spatial_size) % C;
        uint bc = b * C + c;
        float v = float(x[tid]);
        float s = float(scale[bc]);
        float sh = float(shift[bc]);
        output[tid] = half(v * (1.0f + s) + sh);
    }

    // Channel scale-shift: out = x * scale[c] + shift[c]
    // x: [B, C, ...spatial], scale/shift: [C] (broadcast over B and spatial)
    // Grid: total elements
    kernel void channel_scale_shift(
        device const half* x [[buffer(0)]],
        device const half* scale [[buffer(1)]],
        device const half* shift [[buffer(2)]],
        device half* output [[buffer(3)]],
        constant uint& C [[buffer(4)]],
        constant uint& spatial_size [[buffer(5)]],
        uint tid [[thread_position_in_grid]])
    {
        uint c = (tid / spatial_size) % C;
        float v = float(x[tid]);
        float s = float(scale[c]);
        float sh = float(shift[c]);
        output[tid] = half(v * s + sh);
    }

    // Causal pad: repeat first temporal frame. Prepends `pad` copies of frame 0.
    // Input: [B, C, D, H, W], Output: [B, C, D+pad, H, W]
    // Grid: total output elements
    kernel void causal_pad_repeat(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint& C [[buffer(2)]],
        constant uint& D [[buffer(3)]],
        constant uint& H [[buffer(4)]],
        constant uint& W [[buffer(5)]],
        constant uint& pad [[buffer(6)]],
        uint tid [[thread_position_in_grid]])
    {
        uint HW = H * W;
        uint D_out = D + pad;
        uint total_per_bc = D_out * HW;
        uint bc = tid / total_per_bc;
        uint rem = tid % total_per_bc;
        uint od = rem / HW;
        uint hw = rem % HW;

        // Map output depth to input depth: first 'pad' frames -> frame 0
        uint id = (od < pad) ? 0 : (od - pad);
        uint in_idx = bc * D * HW + id * HW + hw;
        output[tid] = input[in_idx];
    }

    // Pad by repeating LAST frame at the end. Input [B*C, D, H, W], output [B*C, D+pad, H, W].
    // Grid: total output elements (B*C * (D+pad) * H * W)
    kernel void causal_pad_back(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint& C [[buffer(2)]],
        constant uint& D [[buffer(3)]],
        constant uint& H [[buffer(4)]],
        constant uint& W [[buffer(5)]],
        constant uint& pad [[buffer(6)]],
        uint tid [[thread_position_in_grid]])
    {
        uint HW = H * W;
        uint D_out = D + pad;
        uint total_per_bc = D_out * HW;
        uint bc = tid / total_per_bc;
        uint rem = tid % total_per_bc;
        uint od = rem / HW;
        uint hw = rem % HW;

        // Map output depth to input depth: frames beyond D-1 -> last frame
        uint id = (od >= D) ? (D - 1) : od;
        uint in_idx = bc * D * HW + id * HW + hw;
        output[tid] = input[in_idx];
    }

    // Unpatchify 3D: [B, C*p*p, F, H, W] -> [B, C, F, H*p, W*p]
    // (patch_size_t=1 case, spatial only)
    // Grid: total output elements
    kernel void unpatchify_3d(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint* params [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        uint C_out = params[0]; // output channels (3)
        uint F = params[1];     // frames
        uint H_out = params[2]; // output height = H_in * p
        uint W_out = params[3]; // output width = W_in * p
        uint p = params[4];     // patch size
        uint B = params[5];

        uint H_in = H_out / p, W_in = W_out / p;
        uint C_in = C_out * p * p;  // input channels
        uint out_spatial = F * H_out * W_out;
        uint total = B * C_out * out_spatial;
        if (tid >= total) return;

        uint rem = tid;
        uint b = rem / (C_out * out_spatial); rem %= (C_out * out_spatial);
        uint c = rem / out_spatial; rem %= out_spatial;
        uint f = rem / (H_out * W_out); rem %= (H_out * W_out);
        uint oh = rem / W_out;
        uint ow = rem % W_out;

        // Reverse: b (c p r q) f h w -> b c (f p_t) (h q) (w r) with p_t=1
        // Input layout: b (c r q) f h w  where r,q = patch indices
        uint ih = oh / p;  // q index
        uint q = oh % p;
        uint iw = ow / p;  // r index
        uint r = ow % p;

        // Input channel = c * p * p + r * p + q
        uint ic = c * p * p + r * p + q;
        uint in_idx = b * C_in * F * H_in * W_in + ic * F * H_in * W_in + f * H_in * W_in + ih * W_in + iw;
        output[tid] = input[in_idx];
    }

    // Timestep embedding (sinusoidal): produces [B, dim] from scalar timestep.
    // half_dim frequencies, sin/cos interleaved.
    // Grid: (dim, B)
    kernel void timestep_embedding(
        device const float* timesteps [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint& dim [[buffer(2)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint i = tid.x;
        uint b = tid.y;
        if (i >= dim) return;

        uint half_dim = dim / 2;
        float t = timesteps[b];
        if (i < half_dim) {
            // flip_sin_to_cos=True: first half is cos
            float freq = exp(-log(10000.0f) * float(i) / float(half_dim));
            output[b * dim + i] = half(cos(t * freq));
        } else {
            // second half is sin
            float freq = exp(-log(10000.0f) * float(i - half_dim) / float(half_dim));
            output[b * dim + i] = half(sin(t * freq));
        }
    }

    // Linear (dense) layer: out = x @ W^T + bias, all f16 with f32 accumulation
    // x: [B, K], W: [N, K], bias: [N], out: [B, N]
    // Grid: (N, B)
    kernel void linear_f16(
        device const half* x [[buffer(0)]],
        device const half* weight [[buffer(1)]],
        device const half* lin_bias [[buffer(2)]],
        device half* output [[buffer(3)]],
        constant uint& K [[buffer(4)]],
        constant uint& N [[buffer(5)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint n = tid.x;
        uint b = tid.y;
        if (n >= N) return;

        float sum = float(lin_bias[n]);
        for (uint k = 0; k < K; k++) {
            sum += float(x[b * K + k]) * float(weight[n * K + k]);
        }
        output[b * N + n] = half(sum);
    }

    // Gather input for one kernel position: for each output position m,
    // extract input[b, :, id, ih, iw] → row of [M, C_in] matrix.
    // id = od*sD+fd-pD, ih = oh*sH+fh-pH, iw = ow*sW+fw-pW.
    // Grid: (C_in, M)
    // params: [C_in, D_in, H_in, W_in, D_out, H_out, W_out, fd, fh, fw, sD, sH, sW, pD, pH, pW]
    kernel void gather_input_3d(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint* params [[buffer(2)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint C_in = params[0];
        uint D_in = params[1], H_in = params[2], W_in = params[3];
        uint D_out = params[4], H_out = params[5], W_out = params[6];
        uint fd = params[7], fh = params[8], fw = params[9];
        uint sD = params[10], sH = params[11], sW = params[12];
        uint pD = params[13], pH = params[14], pW = params[15];

        uint rowOff = params[16];

        uint ic = tid.x;
        uint m_local = tid.y;
        if (ic >= C_in) return;

        uint m = rowOff + m_local;
        uint spatial_out = D_out * H_out * W_out;
        uint b = m / spatial_out;
        uint rem = m % spatial_out;
        uint od = rem / (H_out * W_out);
        uint rem2 = rem % (H_out * W_out);
        uint oh = rem2 / W_out;
        uint ow = rem2 % W_out;

        int id = int(od * sD + fd) - int(pD);
        int ih = int(oh * sH + fh) - int(pH);
        int iw = int(ow * sW + fw) - int(pW);

        half val = 0;
        if (id >= 0 && uint(id) < D_in && ih >= 0 && uint(ih) < H_in && iw >= 0 && uint(iw) < W_in) {
            uint HW_in = H_in * W_in;
            val = input[b * C_in * D_in * HW_in + ic * D_in * HW_in + uint(id) * HW_in + uint(ih) * W_in + uint(iw)];
        }
        output[m_local * C_in + ic] = val;
    }

    // Extract weight slice for one kernel position.
    // Weight: [C_out, C_in, kD, kH, kW] → extract [:, :, fd, fh, fw] → [C_out, C_in] contiguous.
    // weight[oc][ic][p] where p = kPos, stride between ic = kSize.
    // params: [C_in, kSize, kPos]
    // Grid: (C_in, C_out)
    kernel void extract_weight_slice(
        device const half* weight [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint* params [[buffer(2)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint C_in = params[0];
        uint kSize = params[1];
        uint kPos = params[2];

        uint ic = tid.x;
        uint oc = tid.y;
        if (ic >= C_in) return;

        // weight layout: [C_out, C_in, kSize] row-major
        output[oc * C_in + ic] = weight[oc * C_in * kSize + ic * kSize + kPos];
    }

    // Accumulate GEMM output into NCHW conv output.
    // gemm_out: [M, C_out] row-major. out: [B, C_out, spatial] NCHW.
    // If is_first: initialize output to bias (or gemm_out + bias).
    // Otherwise: add gemm_out to existing output.
    // params: [C_out, spatial, is_first, is_unused]
    // Grid: M * C_out
    kernel void accumulate_conv_output(
        device const half* gemm_out [[buffer(0)]],
        device const half* bias_data [[buffer(1)]],
        device half* output [[buffer(2)]],
        constant uint* params [[buffer(3)]],
        uint tid [[thread_position_in_grid]])
    {
        uint C_out = params[0];
        uint spatial = params[1];
        uint is_first = params[2];
        uint rowOff = params[3];

        uint m_local = tid / C_out;
        uint c = tid % C_out;
        uint m = rowOff + m_local;
        uint b = m / spatial;
        uint s = m % spatial;

        uint out_idx = b * C_out * spatial + c * spatial + s;
        float g = float(gemm_out[m_local * C_out + c]);

        if (is_first) {
            output[out_idx] = half(g + float(bias_data[c]));
        } else {
            output[out_idx] = half(float(output[out_idx]) + g);
        }
    }

    // AdaLN combine (4 outputs): table[4,C] + tsEmb[B,4C] → 4 separate [B,C] outputs
    // Grid: B * C threads
    kernel void ada_ln_combine4(
        device const half* table [[buffer(0)]],
        device const half* tsEmb [[buffer(1)]],
        device half* out0 [[buffer(2)]],
        device half* out1 [[buffer(3)]],
        device half* out2 [[buffer(4)]],
        device half* out3 [[buffer(5)]],
        constant uint& C [[buffer(6)]],
        uint tid [[thread_position_in_grid]])
    {
        uint b = tid / C;
        uint c = tid % C;
        uint base = b * 4 * C;
        out0[tid] = half(float(table[c]) + float(tsEmb[base + c]));
        out1[tid] = half(float(table[C + c]) + float(tsEmb[base + C + c]));
        out2[tid] = half(float(table[2*C + c]) + float(tsEmb[base + 2*C + c]));
        out3[tid] = half(float(table[3*C + c]) + float(tsEmb[base + 3*C + c]));
    }

    // AdaLN combine (2 outputs): table[2,C] + tsEmb[B,2C] → 2 separate [B,C] outputs
    // Grid: B * C threads
    kernel void ada_ln_combine2(
        device const half* table [[buffer(0)]],
        device const half* tsEmb [[buffer(1)]],
        device half* out0 [[buffer(2)]],
        device half* out1 [[buffer(3)]],
        constant uint& C [[buffer(4)]],
        uint tid [[thread_position_in_grid]])
    {
        uint b = tid / C;
        uint c = tid % C;
        uint base = b * 2 * C;
        out0[tid] = half(float(table[c]) + float(tsEmb[base + c]));
        out1[tid] = half(float(table[C + c]) + float(tsEmb[base + C + c]));
    }

    // Repeat channels: [B, C, spatial] → [B, C*repeats, spatial]
    // Each output element maps to a source element with channel index wrapped.
    // Grid: total output elements (B * C * repeats * spatial)
    kernel void repeat_channels(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint& C [[buffer(2)]],
        constant uint& spatial [[buffer(3)]],
        constant uint& repeats [[buffer(4)]],
        uint tid [[thread_position_in_grid]])
    {
        uint outC = C * repeats;
        uint bc_out = tid / spatial;
        uint s = tid % spatial;
        uint b = bc_out / outC;
        uint c_out = bc_out % outC;
        uint c_in = c_out % C;
        output[tid] = input[b * C * spatial + c_in * spatial + s];
    }

    // Temporal slice: x[:, :, from:, :, :] — copy with offset along depth dimension.
    // Input: [B*C, D, H*W], Output: [B*C, D_out, H*W] where D_out = D - from
    // Grid: total output elements (B * C * D_out * H * W)
    kernel void temporal_slice(
        device const half* input [[buffer(0)]],
        device half* output [[buffer(1)]],
        constant uint& D [[buffer(2)]],
        constant uint& D_out [[buffer(3)]],
        constant uint& HW [[buffer(4)]],
        constant uint& from_offset [[buffer(5)]],
        uint tid [[thread_position_in_grid]])
    {
        uint bc = tid / (D_out * HW);
        uint rem = tid % (D_out * HW);
        uint d_out = rem / HW;
        uint hw = rem % HW;
        output[tid] = input[bc * D * HW + (from_offset + d_out) * HW + hw];
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
