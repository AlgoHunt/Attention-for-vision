#include <ATen/ATen.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>




static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H, int W)
{
  return h >= 0 && h < H && w >= 0 && w < W;
}
/*
static __forceinline__ __device__
  bool clip_2d(float x0, float y0,float x1, float y1 , int H, int W) {
    //return h >= 0 && h < H && w >= 0 && w < W;
    return 0;
  }
*/

// ================================

template <typename T>
__global__ void ia_forward_kernel(const T *t, const T *f, T *weight, int num, int chn, int height, int width, int factor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int w_num = width/factor;
  int h_num = height/factor;
  int sampleNum =  w_num * h_num;
 // int d3 = w_sum*h_sum;

  //int sampleNum =  factor * factor;
  int z = blockIdx.z;

  if (x < width && y < height && z < sampleNum) {
    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {


        int ix = x + z % w_num - w_num / 2;
        int iy = y + z / w_num - h_num / 2;
        if (within_bounds_2d(iy, ix, height, width)){
          float _t = t[((batch * chn + plane) * height + y)*width + x];
          float _f = f[((batch * chn + plane) * height + iy)*width + ix];
          weight[((batch * sampleNum + z) * height + y) * width + x] += _t*_f;

           }
        }
      }
    }
}






// w : n, s, h, w
//
template <typename T>
__global__ void ia_backward_kernel_all(const T *dw, const T *t, const T *f, T *dt, T *df,
                                       int num, int chn, int height, int width, int factor)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int ix, iy;
  int plane = blockIdx.z;

  int w_num = width/factor;
  int h_num = height/factor;
  int sampleNum =  w_num * h_num;

  //int sampleNum = factor * factor;
  if (x < width && y < height && plane < chn)
  {
    for (int batch = 0; batch < num; ++batch)
    {

      for (int idx = 0; idx < sampleNum; ++idx)
      {

        ix = x + idx % w_num - w_num / 2;
        iy = y + idx / w_num - h_num / 2;

        if (within_bounds_2d(iy, ix, height, width))
        {
          float _dw = dw[((((batch * sampleNum) + idx) * height) + y) * width + x];
          float _t = t[((batch * chn + plane) * height + y)*width + x];
          float _f = f[((batch * chn + plane) * height + iy)*width + ix];
          dt[((batch * chn + plane) * height + y) * width + x] += _dw * _f;
          atomicAdd(df + ((batch * chn + plane) * height + iy) * width + ix,
                    static_cast<float>(_dw * _t));
        }
      }
    }
  }
}


template <typename T>
__global__ void ia_map_forward_kernel(const T *weight, const T *g, T *out,
                                      int num, int chn, int height, int width, int factor)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int plane = blockIdx.z;

  int w_num = width/factor;
  int h_num = height/factor;
  int sampleNum =  w_num * h_num;

  //int sampleNum = factor * factor;
  if (x < width && y < height && plane < chn)
  {
    for (int batch = 0; batch < num; ++batch)
    {
      for (int idx = 0; idx < sampleNum; ++idx)
      {

        int ix = x + idx % w_num - w_num / 2;
        int iy = y + idx / w_num - h_num / 2;

        if (within_bounds_2d(iy, ix, height, width))
        {
          float _g = g[((batch * chn + plane) * height + iy) * width + ix];
          float _w = weight[(((batch * sampleNum + idx) * height) + y) * width + x];
          out[(((batch * chn + plane) * height) + y) * width + x] += _g * _w;
        }
      }
    }
  }
}

template <typename T>
__global__ void ia_map_backward_kernel_w(const T *dout, const T *weight, const T *g, T *dw,
                                int num, int chn, int height, int width, int factor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  //int sampleNum = factor*factor;
  int z = blockIdx.z;


  int w_num = width/factor;
  int h_num = height/factor;
  int sampleNum =  w_num * h_num;

  if (x < width && y < height && z < sampleNum) {

    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        float _dout = dout[(batch * chn + plane) * sp + y*width + x];

        int ix = x + z % w_num - w_num / 2;
        int iy = y + z / w_num - h_num / 2;
        if (within_bounds_2d(iy, ix, height, width)){
          float _g = g[(batch * chn + plane) * sp + iy*width + ix];
          dw[((batch * sampleNum + z)*height + y)*width + x] += _dout * _g;
        }
        }
      }
    }
  }


template <typename T>
__global__ void ia_map_backward_kernel_g(const T *dout, const T *weight, const T *g, T *dg,
                                int num, int chn, int height, int width,int factor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int sampleNum = factor*factor;
  int plane = blockIdx.z;

  int w_num = width/factor;
  int h_num = height/factor;
  int sampleNum =  w_num * h_num;


  if (x < width && y < height && plane < chn) {

    for (int batch = 0; batch < num; ++batch) {
      for (int idx = 0; idx < sampleNum; ++idx) {


        int ix = x + idx % w_num - w_num / 2 - w_num % 2 + 1;
        int iy = y + idx / w_num - h_num / 2 - h_num % 2 + 1;

        if (within_bounds_2d(iy, ix, height, width)){


            int reverse_idx = sampleNum - idx-1;

            float _dout = dout[((batch * chn + plane) * height + iy)*width + ix];

            float _w = weight[((((batch * sampleNum) + reverse_idx) * height) + iy) * width + ix];

            dg[((batch * chn + plane) * height + y)*width + x] +=  _dout * _w;


        }
      }
    }
  }
}

/*
 * Implementations
 */
int _ia_forward_cuda(int N, int C, int H, int W, int factor, at::Tensor t, at::Tensor f, at::Tensor weight)
{
  //Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;

  int w_num = W/factor;
  int h_num = H/factor;
  int d3 = w_num*h_num;

  //int d3 = factor*factor;
  dim3 blocks(d1, d2, d3);
  //printf("%d %d %d %d %d \n",N,C,H,W,factor*factor);
  ia_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(t.data<float>(), f.data<float>(), weight.data<float>(), N, C, H, W, factor);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

int _ia_backward_cuda(int N, int C, int H, int W, int factor, const at::Tensor dw, const at::Tensor t, const at::Tensor f, at::Tensor dt, at::Tensor df)
{
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;
  int d3 = C;
  dim3 blocks(d1, d2, d3);
  // printf("%f\n", dw[0]);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ia_backward_kernel_all<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dw.data<float>(), t.data<float>(), f.data<float>(), dt.data<float>(), df.data<float>(), N, C, H, W, factor);

  // Check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

int _ia_map_forward_cuda(int N, int C, int H, int W, int factor, const at::Tensor weight, const at::Tensor g, at::Tensor out)
{
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;
  int d3 = C;
  dim3 blocks(d1, d2, d3);
  ia_map_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(weight.data<float>(),
                                                                                  g.data<float>(), out.data<float>(), 
                                                                                  N, C, H, W, factor);

  // Check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

int _ia_map_backward_cuda(int N, int C, int H, int W, int factor, const at::Tensor dout, const at::Tensor weight,
                          const at::Tensor g, at::Tensor dw, at::Tensor dg)
{
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;
  int w_num = W/factor;
  int h_num = H/factor;
  int d3 = w_num*h_num;



  dim3 blocks(d1, d2, d3);
  ia_map_backward_kernel_w<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dout.data<float>(),
                                                                                   weight.data<float>(), 
                                                                                   g.data<float>(), dw.data<float>(),
                                                                                     N, C, H, W, factor);
  d3 = C;
  blocks = dim3(d1, d2, d3);
  ia_map_backward_kernel_g<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dout.data<float>(),
                                                                                   weight.data<float>(),
                                                                                   g.data<float>(),
                                                                                    dg.data<float>(),
                                                                                    N, C, H, W, factor);

  // Check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}
