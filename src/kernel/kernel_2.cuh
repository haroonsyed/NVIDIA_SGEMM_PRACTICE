#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // 申请共享内存空间
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {
        // 缓存A_tile和B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

__global__ void haroon_mysgemm_v2(float *mat1_buffer, int mat1_rows, int mat1_cols, float *mat2_buffer, int mat2_rows, int mat2_cols, float *out_buffer, int out_rows, int out_cols) {
    const int block_dim = 32;
    const int block_area = block_dim * block_dim;

    // Block tiling with shared memory
    __shared__ float s_mat1[block_area];
    __shared__ float s_mat2[block_area];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    int mat1_block_pos = block_row * block_dim * mat1_cols;
    int mat2_block_pos = block_col * block_dim;
    int out_block_pos = block_row * block_dim * out_cols + block_col * block_dim;

    // So within our block we are gonna figure out this thread's position
    // int thread_row = threadIdx.x / block_dim;
    // int thread_col = threadIdx.x % block_dim;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int out_row = block_row * block_dim + thread_row;
    int out_col = block_col * block_dim + thread_col;
    if (out_row >= out_rows || out_col >= out_cols) {
        return;
    }

    float weighted_sum = 0.0;
    int common_partial_block = mat1_cols % block_dim;
    int common_in_block = mat1_cols - common_partial_block;
    for (int k = 0; k < common_in_block; k += block_dim) {
        s_mat1[thread_row * block_dim + thread_col] = mat1_buffer[mat1_block_pos + thread_row * mat1_cols + thread_col];
        s_mat2[thread_row * block_dim + thread_col] = mat2_buffer[mat2_block_pos + thread_row * mat2_cols + thread_col];
        __syncthreads();

        mat1_block_pos += block_dim;
        mat2_block_pos += block_dim * mat2_cols;
        for (int i = 0; i < block_dim; i++) {
            weighted_sum += s_mat1[thread_row * block_dim + i] * s_mat2[i * block_dim + thread_col];
        }
        __syncthreads();
    }

    // Handle partial block case
    s_mat1[thread_row * block_dim + thread_col] = mat1_buffer[mat1_block_pos + thread_row * mat1_cols + thread_col];
    s_mat2[thread_row * block_dim + thread_col] = mat2_buffer[mat2_block_pos + thread_row * mat2_cols + thread_col];
    __syncthreads();

    mat1_block_pos += block_dim;
    mat2_block_pos += block_dim * mat2_cols;
    for (int i = 0; i < common_partial_block; i++) {
        weighted_sum += s_mat1[thread_row * block_dim + i] * s_mat2[i * block_dim + thread_col];
    }

    out_buffer[out_block_pos + (thread_row * out_cols) + thread_col] = weighted_sum;
}