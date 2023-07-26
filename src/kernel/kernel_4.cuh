#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

template <const int BM,
          const int BN,
          const int BK,
          const int TM,
          const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;  // 一个线程负责计算block中TM*TN个元素

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.};  // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll  // 循环展开，增加指令并行度
            for (int j = 0; j < TM; j++) {
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }
}

// block_M is rows in mat1 shared block
// block_N is cols in mat2 shared block
// block_k is shared dimensions for shared block.
// The thread will calculate block_k * block_k results (So now a 2d version of v3)
// For this to work we want the shared dimension block_K to be extremely smaller than block_M and block_N
// This way, multiple threads reuse sections from mat1 and mat2 ,with more output work
// Example: bK is 8 while bM and bN are 128. Output is a 128x128 area.
//          So you can spin up 256 threads per block. They load vram->shared
//          Then each thread can work on 8x8 pieces of the output 128x128 area (128x128/64 = 256)
//          You might be wondering why not 512 threads like previously?
//          Well that increases the mem requirements per block, reducing occupancy.
template <const int block_M, const int block_N, const int block_K>
__global__ void haroon_mysgemm_v4(int M, int N, int K, float *mat1_buffer, float *mat2_buffer, float *out_buffer) {
    // 2D Block tiling with shared memory
    __shared__ float s_mat1[block_M * block_K];
    __shared__ float s_mat2[block_K * block_N];

    float thread_results[block_K * block_K] = {0.0};

    float cache_mat1[block_K] = {0.0};
    float cache_mat2[block_K] = {0.0};

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Output within block details
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int out_block_row = tid / (block_M / block_K);
    const int out_block_col = tid % (block_N / block_K);

    const int num_elements_to_load = (block_M * block_K) / (blockDim.x * blockDim.y);

    // outer loop over block tiles
    for (uint common_block = 0; common_block < K; common_block += block_K) {
        for (int i = 0; i < num_elements_to_load; i++) {
            // Used to track if out of bounds
            const int mat1_load_index_row = block_row * block_M + threadIdx.x + (i * blockDim.x);
            const int mat2_load_index_col = block_col * block_N + threadIdx.x + (i * blockDim.x);
            const int mat_common_index = common_block + threadIdx.y;
            const bool exceeded_mat1_row = mat1_load_index_row >= M;
            const bool exceeded_mat2_col = mat2_load_index_col >= N;

            const int within_mat1 = (int)!(exceeded_mat1_row || mat_common_index >= K);
            const int within_mat2 = (int)!(mat_common_index >= K || exceeded_mat2_col);
            int mat1_load_index = mat1_load_index_row * K + mat_common_index;
            int mat2_load_index = mat_common_index * N + mat2_load_index_col;

            // Prevent loading OOB
            mat1_load_index *= within_mat1;
            mat2_load_index *= within_mat2;

            s_mat1[(threadIdx.x + (i * blockDim.x)) * block_K + threadIdx.y] =
                mat1_buffer[mat1_load_index] * within_mat1;

            s_mat2[threadIdx.y * block_N + (threadIdx.x + i * blockDim.x)] =
                mat2_buffer[mat2_load_index] * within_mat2;
        }
        __syncthreads();

        // Go through common dimensions of block (across row of mat1 and down col of mat2)
        for (int block_common_index = 0; block_common_index < block_K; block_common_index++) {
            // Cache rows and cols  on thread
            for (int i = 0; i < block_K; i++) {
                cache_mat1[i] = s_mat1[(out_block_row * block_K + i) * block_K + block_common_index];
            }
            for (int i = 0; i < block_K; i++) {
                cache_mat2[i] = s_mat2[(block_common_index * block_N) + (out_block_col * block_K + i)];
            }

            // Now this thread will accumulate the block_K x block_K results from shared memory
            for (int result_index_row = 0; result_index_row < block_K; result_index_row++) {
                for (int result_index_col = 0; result_index_col < block_K; result_index_col++) {
                    thread_results[result_index_row * block_K + result_index_col] +=
                        cache_mat1[result_index_row] *
                        cache_mat2[result_index_col];
                }
            }
        }
        __syncthreads();
    }

    // Write results with bounds checking
    const int out_index_row = block_row * block_M + out_block_row * block_K;
    const int out_index_col = block_col * block_N + out_block_col * block_K;

    for (int i = 0; i < block_K; i++) {
        for (int j = 0; j < block_K; j++) {
            if (out_index_row + i < M && out_index_col + j < N) {
                out_buffer[(out_index_row + i) * N + out_index_col + j] = thread_results[i * block_K + j];
            }
        }
    }
}