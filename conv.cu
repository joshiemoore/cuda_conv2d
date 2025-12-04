#include <cuda_runtime.h>
#include <stdio.h>

#include "conv.h"

__constant__ char _kernel[KERNEL_ROWS * KERNEL_COLS];

__global__ void conv2d_kernel(const unsigned char* input, unsigned char* output, int rows, int cols) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;

  float sum[3] = { 0.0f };
  for (int kr = -KERNEL_ROWS / 2; kr <= KERNEL_ROWS / 2; kr++) {
    for (int kc = -KERNEL_COLS / 2; kc <= KERNEL_COLS / 2; kc++) {
      const int pidx = (row + kr) * cols * 3 + (col + kc) * 3;
      for (int chan = 0; chan < 3; chan++) {
        const float pval = (row+kr >= 0 && row+kr < rows && col+kc >= 0 && col+kc < cols) ? (float)(input[pidx + chan]) : 0.0f;
        sum[chan] += pval * (float)_kernel[(kr + KERNEL_ROWS/2) * KERNEL_COLS + (kc + KERNEL_COLS/2)];
      }
    }
  }

  for (int chan = 0; chan < 3; chan++) {
    output[row * cols * 3 + col * 3 + chan] = (unsigned char)(sum[chan]);
  }
}

extern "C" void conv2d(const unsigned char* input, unsigned char* output, int rows, int cols, const char* kernel) {
  const dim3 threads(32, 32);
  const dim3 blocks(
    (cols + threads.x - 1) / threads.x,
    (rows + threads.y - 1) / threads.y
  );

  unsigned char* input_d;
  unsigned char* output_d;
  cudaMalloc(&input_d, rows * cols * 3);
  cudaMalloc(&output_d, rows * cols * 3);

  cudaMemcpy(input_d, input, rows * cols * 3, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(_kernel, kernel, KERNEL_ROWS * KERNEL_COLS);

  conv2d_kernel<<<blocks, threads>>>(input_d, output_d, rows, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(output, output_d, rows * cols * 3, cudaMemcpyDeviceToHost);

  cudaFree(output_d);
  cudaFree(input_d);
}
