code = """
#include <cupy/carray.cuh>  // for float16
#define CUDA_KERNEL_LOOP(i, n)                          for (int i = blockIdx.x * blockDim.x + threadIdx.x;       i < (n);                                             i += blockDim.x * gridDim.x)

extern "C"
__global__ void aggregation_zeropad_forward_kernel(
const float16* bottom_data, const float16* weight_data, float16* top_data) {
  CUDA_KERNEL_LOOP(index, 524288) {
    const int n = index / 1 / 256 / 32 / 32;
    const int head = (index / 32 / 32 / 256) % 1;
    const int c = (index / 32 / 32) % 256;
    const int h = (index / 32) % 32;
    const int w = index % 32;

    float16 value = 0;
    for (int kh = 0; kh < 3; ++kh) {
      for (int kw = 0; kw < 3; ++kw) {
        const int h_in = -1 + h * 1 + kh * 1;
        const int w_in = -1 + w * 1 + kw * 1;
        if ((h_in >= 0) && (h_in < 32) && (w_in >= 0) && (w_in < 32)) {
          const int offset_bottom = ((n * 256 + c) * 32 + h_in) * 32 + w_in;
          const int offset_weight = (((n * 1 + head) * 32 + c % 32) * 3 * 3 + (kh * 3 + kw)) * 32 * 32 + h * 32 + w;
          value += weight_data[offset_weight] * bottom_data[offset_bottom];
        }
      }
    }
    top_data[index] = value;
  }
}"""
import cupy

if __name__ == '__main__':
    kernel_code = cupy.cuda.compile_with_cache(code, options=(
        "-I/home/caiqi/anaconda3/lib/python3.7/site-packages/cupy/_core/include",))