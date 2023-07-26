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
__global__ void haroon_mysgemm_v4(int M, int N, int K, float *__restrict__ mat1_buffer, float *__restrict__ mat2_buffer, float *__restrict__ out_buffer) {
    // 2D Block tiling with shared memory
    __shared__ float s_mat1[block_M * block_K];
    __shared__ float s_mat2[block_K * block_N];

    float thread_results[block_K * block_K] = {0.0};

    // float cache_mat1[block_K] = {0.0};
    // float cache_mat2[block_K] = {0.0};

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Output within block details
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int out_block_row = tid / (block_M / block_K);
    const int out_block_col = tid % (block_N / block_K);

    const int num_threads_per_block = blockDim.x * blockDim.y;
    const int num_elements_to_load = (block_M * block_K) / num_threads_per_block;

    const int stride_mat1 = num_threads_per_block / block_K;
    const int stride_mat2 = num_threads_per_block / block_N;

    int mat1_pos = block_row * block_M * K;
    int mat2_pos = block_col * block_N;

// outer loop over block tiles
#pragma unroll
    for (int common_block = 0; common_block < K; common_block += block_K) {
#pragma unroll 4
        for (int i = 0; i < num_elements_to_load; i++) {
            const int mat1_row_within_block = (threadIdx.x + stride_mat1 * i);
            const int mat1_col_within_block = threadIdx.y;
            const int mat2_row_within_block = (threadIdx.y / num_elements_to_load) + i * stride_mat2;
            const int mat2_col_within_block = (threadIdx.y % num_elements_to_load) * blockDim.x + threadIdx.x;

            const int mat1_load_index_row = block_row * block_M + mat1_row_within_block;
            const int mat1_load_index_col = common_block + mat1_col_within_block;
            const int mat2_load_index_row = common_block + mat2_row_within_block;
            const int mat2_load_index_col = block_col * block_N + mat2_col_within_block;

            const bool exceeded_mat1_row = mat1_load_index_row >= M;
            const bool exceeded_mat1_col = mat1_load_index_col >= K;
            const bool exceeded_mat2_row = mat2_load_index_row >= K;
            const bool exceeded_mat2_col = mat2_load_index_col >= N;

            const int within_mat1 = (int)!(exceeded_mat1_row || exceeded_mat1_col);
            const int within_mat2 = (int)!(exceeded_mat2_row || exceeded_mat2_col);
            int mat1_load_index = mat1_pos + mat1_row_within_block * K + mat1_col_within_block;
            int mat2_load_index = mat2_pos + mat2_row_within_block * K + mat2_col_within_block;

            mat1_load_index *= within_mat1;
            mat2_load_index *= within_mat2;

            s_mat1[mat1_row_within_block * block_K + mat1_col_within_block] =
                mat1_buffer[mat1_load_index] * within_mat1;
            s_mat2[mat2_row_within_block * block_N + mat2_col_within_block] =
                mat2_buffer[mat2_load_index] * within_mat2;
        }

        mat1_pos += block_K;
        mat2_pos += block_K * N;

        __syncthreads();

        // Go through common dimensions of block (across row of mat1 and down col of mat2)
#pragma unroll 8
        for (int block_common_index = 0; block_common_index < block_K; block_common_index++) {
            //             // Cache rows and cols  on thread
            // #pragma unroll 8
            //             for (int i = 0; i < block_K; i++) {
            //                 cache_mat1[i] = s_mat1[(out_block_row * block_K + i) * block_K + block_common_index];
            //             }
            // #pragma unroll 8
            //             for (int i = 0; i < block_K; i++) {
            //                 cache_mat2[i] = s_mat2[(block_common_index * block_N) + (out_block_col * block_K + i)];
            //             }

            // Now this thread will accumulate the block_K x block_K results from shared memory
#pragma unroll 8
            for (int result_index_row = 0; result_index_row < block_K; result_index_row++) {
#pragma unroll 8
                for (int result_index_col = 0; result_index_col < block_K; result_index_col++) {
                    thread_results[result_index_row * block_K + result_index_col] +=
                        s_mat1[(out_block_row * block_K + result_index_row) * block_K + block_common_index] *
                        s_mat2[(block_common_index * block_N) + (out_block_col * block_K + result_index_col)];
                    // cache_mat1[result_index_row] *
                    // cache_mat2[result_index_col];
                }
            }
        }
        __syncthreads();
    }

    // Write results with bounds checking
    const int out_index_row = block_row * block_M + out_block_row * block_K;
    const int out_index_col = block_col * block_N + out_block_col * block_K;

#pragma unroll
    for (int i = 0; i < block_K; i++) {
#pragma unroll
        for (int j = 0; j < block_K; j++) {
            if (out_index_row + i < M && out_index_col + j < N) {
                out_buffer[(out_index_row + i) * N + out_index_col + j] = thread_results[i * block_K + j];
            }
        }
    }
}