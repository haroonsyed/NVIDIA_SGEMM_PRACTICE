#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ __launch_bounds__(1024) void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局x
    int gy = blockIdx.y * blockDim.y + threadIdx.y;  // 全局y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx];  // 两次全局内存访问和一次FMA（累加乘）
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

__global__ void matrix_multiply_kernel_1(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    // Go by col row instead of row col. Enabled memory coalescing
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= out_rows || col >= out_cols) {
        return;
    }

    float weighted_sum = 0.0;
    for (int common = 0; common < mat1_cols; common++) {
        int mat1_index = mat1_cols * row + common;
        int mat2_index = mat2_cols * common + col;
        weighted_sum += mat1_buffer[mat1_index] * mat2_buffer[mat2_index];
    }
    int output_index = row * out_cols + col;
    out_buffer[output_index] = weighted_sum;
}