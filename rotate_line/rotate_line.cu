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
__global__ void forward_kernel(const T *t, const T *f, T *weight, T *pixel_position, int num, int chn, int height, int width, int NumSample)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z;
  if (x < width && y < height && i < NumSample)
  {
    for (int batch = 0; batch < num; ++batch)
    {
      for (int plane = 0; plane < chn; ++plane)
      {
        float _t = t[((batch * chn + plane) * height + y) * width + x];

        float pos_x = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 0];
        float pos_y = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 1];
        float in_x = ((pos_x + 1) / 2) * (width - 1);
        float in_y = ((pos_y + 1) / 2) * (height - 1);
        // get y coords
        const int top_y_index = floor(in_y);
        const int bottom_y_index = ceil(in_y);

        // get x coords
        const int left_x_index = floor(in_x);
        const int right_x_index = ceil(in_x);

        const float y_lerp = in_y - top_y_index;
        const float x_lerp = in_x - left_x_index;

        float top_left, top_right, bottom_left, bottom_right;
        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {
          top_left = static_cast<float>(
              f[((batch * chn + plane) * height + top_y_index) * width + left_x_index]);
        }
        else
        {
          top_left = 0;
        }

        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          top_right = (static_cast<float>(
              f[((batch * chn + plane) * height + top_y_index) * width + right_x_index]));
        }
        else
        {
          top_right = 0;
        }

        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          bottom_left = (static_cast<float>(
              f[((batch * chn + plane) * height + bottom_y_index) * width + left_x_index]));
        }
        else
        {
          bottom_left = 0;
        }

        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          bottom_right = (static_cast<float>(
              f[((batch * chn + plane) * height + bottom_y_index) * width + right_x_index]));
        }
        else
        {
          bottom_right = 0;
        }

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

        float _f = top + (bottom - top) * y_lerp;

        weight[((batch * NumSample + i) * height + y) * width + x] += _t * _f;
      }
    }
  }
}


template <typename T>
__global__ void backward_kernel_all(const T *dw, const T *t, const T *f, const T *pixel_position, T *dt, T *df, T *dp,
                                       int num, int chn, int height, int width, int NumSample)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn)
  {
    for (int batch = 0; batch < num; ++batch)
    {
      for (int i = 0; i < NumSample; ++i)
      {
        float pos_x = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 0];
        float pos_y = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 1];
        float in_x = ((pos_x + 1) / 2) * (width - 1);
        float in_y = ((pos_y + 1) / 2) * (height - 1);

        // get y coords
        const int top_y_index = floor(in_y);
        const int bottom_y_index = top_y_index + 1;
        const float y_lerp = in_y - top_y_index;

        // get x coords
        const int left_x_index = floor(in_x);
        const int right_x_index = left_x_index + 1;
        const float x_lerp = in_x - left_x_index;

        float top_left, top_right, bottom_left, bottom_right;
        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {
          top_left = static_cast<float>(
              f[((batch * chn + plane) * height + top_y_index) * width + left_x_index]);
        }
        else
        {
          top_left = 0;
        }
        
        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          top_right = (static_cast<float>(
              f[((batch * chn + plane) * height + top_y_index) * width + right_x_index]));
        }
        else
        {
          top_right = 0;
        }

        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          bottom_left = (static_cast<float>(
              f[((batch * chn + plane) * height + bottom_y_index) * width + left_x_index]));
        }
        else
        {
          bottom_left = 0;
        }

        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          bottom_right = (static_cast<float>(
              f[((batch * chn + plane) * height + bottom_y_index) * width + right_x_index]));
        }
        else
        {
          bottom_right = 0;
        }

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

        float _dw = dw[((((batch * NumSample) + i) * height) + y) * width + x];
        float _f = top + (bottom - top) * y_lerp;
        float _t = t[((batch * chn + plane) * height + y) * width + x];

        dt[((batch * chn + plane) * height + y) * width + x] += _dw * _f;

        float _df = _dw * _t;

        const float dtop = (1 - y_lerp) * _df;
        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {
          atomicAdd(df + ((batch * chn + plane) * height + top_y_index) * width + left_x_index,
                    static_cast<float>((1 - x_lerp) * dtop));
        }
        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          atomicAdd(df + ((batch * chn + plane) * height + top_y_index) * width + right_x_index,
                    static_cast<float>(x_lerp * dtop));
        }
        const float dbottom = y_lerp * _df;
        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          atomicAdd(df + ((batch * chn + plane) * height + bottom_y_index) * width + left_x_index,
                    static_cast<float>((1 - x_lerp) * dbottom));
        }
        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          atomicAdd(df + ((batch * chn + plane) * height + bottom_y_index) * width + right_x_index,
                    static_cast<float>(x_lerp * dbottom));
        }

        float _dpx = static_cast<float>(0), _dpy = static_cast<float>(0);

        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {

          _dpx -= top_left * (1 - y_lerp) * _df;
          _dpy -= top_left * (1 - x_lerp) * _df;
        }
        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          _dpx += top_right * (1 - y_lerp) * _df;
          _dpy -= top_right * (x_lerp)*_df;
        }
        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          _dpx -= bottom_left * (y_lerp)*_df;
          _dpy += bottom_left * (1 - x_lerp) * _df;
        }
        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          _dpx += bottom_right * (y_lerp)*_df;
          _dpy += bottom_right * (x_lerp)*_df;
        }
        _dpx = _dpx * (width - 1.f) / 2;
        _dpy = _dpy * (height - 1.f) / 2;

        atomicAdd(dp + (((batch * NumSample + i) * height + y) * width + x) * 2 + 0,
                  static_cast<float>(_dpx));
        atomicAdd(dp + (((batch * NumSample + i) * height + y) * width + x) * 2 + 1,
                  static_cast<float>(_dpy));
      }
    }
  }
}

template <typename T>
__global__ void map_forward_kernel(const T *weight, const T *g, const T *pixel_position, T *out,
                                      int num, int chn, int height, int width, int NumSample)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn)
  {
    for (int batch = 0; batch < num; ++batch)
    {

      for (int i = 0; i < NumSample; ++i)
      {
        //float _g = g[(batch * chn + plane) * sp + y*width + i];

        float pos_x = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 0];
        float pos_y = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 1];
        float in_x = ((pos_x + 1) / 2) * (width - 1);
        float in_y = ((pos_y + 1) / 2) * (height - 1);

        // get y coords
        const int top_y_index = floor(in_y);
        const int bottom_y_index = top_y_index + 1;
        const float y_lerp = in_y - top_y_index;

        // get x coords
        const int left_x_index = floor(in_x);
        const int right_x_index = left_x_index + 1;
        const float x_lerp = in_x - left_x_index;

        float top_left, top_right, bottom_left, bottom_right;
        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {
          top_left = static_cast<float>(
              g[((batch * chn + plane) * height + top_y_index) * width + left_x_index]);
        }
        else
        {
          top_left = 0;
        }

        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          top_right = (static_cast<float>(
              g[((batch * chn + plane) * height + top_y_index) * width + right_x_index]));
        }
        else
        {
          top_right = 0;
        }

        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          bottom_left = (static_cast<float>(
              g[((batch * chn + plane) * height + bottom_y_index) * width + left_x_index]));
        }
        else
        {
          bottom_left = 0;
        }

        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          bottom_right = (static_cast<float>(
              g[((batch * chn + plane) * height + bottom_y_index) * width + right_x_index]));
        }
        else
        {
          bottom_right = 0;
        }
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

        float _g = top + (bottom - top) * y_lerp;
        float _w = weight[(((batch * NumSample + i) * height) + y) * width + x];

        out[(((batch * chn + plane) * height) + y) * width + x] += _g * _w;
      }
    }
  }
}

template <typename T>
__global__ void map_backward_kernel(const T *dout, const T *weight, const T *g, const T *pixel_position, T *dw,
                                       T *dg, T *dp, int num, int chn, int height, int width, int NumSample)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int len = height + width - 1;
  int i = blockIdx.z;

  if (x < width && y < height && i < NumSample)
  {

    for (int batch = 0; batch < num; ++batch)
    {
      for (int plane = 0; plane < chn; ++plane)
      {

        float pos_x = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 0];
        float pos_y = pixel_position[(((batch * NumSample + i) * height + y) * width + x) * 2 + 1];
        float in_x = ((pos_x + 1) / 2) * (width - 1);
        float in_y = ((pos_y + 1) / 2) * (height - 1);

        // get y coords
        const int top_y_index = floor(in_y);
        const int bottom_y_index = top_y_index + 1;
        const float y_lerp = in_y - top_y_index;

        // get x coords
        const int left_x_index = floor(in_x);
        const int right_x_index = left_x_index + 1;
        const float x_lerp = in_x - left_x_index;

        float top_left, top_right, bottom_left, bottom_right;
        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {
          top_left = static_cast<float>(
              g[((batch * chn + plane) * height + top_y_index) * width + left_x_index]);
        }
        else
        {
          top_left = 0;
        }

        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          top_right = (static_cast<float>(
              g[((batch * chn + plane) * height + top_y_index) * width + right_x_index]));
        }
        else
        {
          top_right = 0;
        }

        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          bottom_left = (static_cast<float>(
              g[((batch * chn + plane) * height + bottom_y_index) * width + left_x_index]));
        }
        else
        {
          bottom_left = 0;
        }

        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          bottom_right = (static_cast<float>(
              g[((batch * chn + plane) * height + bottom_y_index) * width + right_x_index]));
        }
        else
        {
          bottom_right = 0;
        }
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

        float _g = top + (bottom - top) * y_lerp;
        float _dout = dout[((batch * chn + plane) * height + y) * width + x];
        float _w = weight[((batch * NumSample + i) * height + y) * width + x];
        float _dg = _dout * _w;

        // backward to dw
        dw[((batch * NumSample + i) * height + y) * width + x] += _dout * _g;

        // backward to dg
        const float dtop = (1 - y_lerp) * _dg;
        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {
          atomicAdd(dg + ((batch * chn + plane) * height + top_y_index) * width + left_x_index,
                    static_cast<float>((1 - x_lerp) * dtop));
        }
        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          atomicAdd(dg + ((batch * chn + plane) * height + top_y_index) * width + right_x_index,
                    static_cast<float>(x_lerp * dtop));
        }
        const float dbottom = y_lerp * _dg;
        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          atomicAdd(dg + ((batch * chn + plane) * height + bottom_y_index) * width + left_x_index,
                    static_cast<float>((1 - x_lerp) * dbottom));
        }
        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          atomicAdd(dg + ((batch * chn + plane) * height + bottom_y_index) * width + right_x_index,
                    static_cast<float>(x_lerp * dbottom));
        }
        // backward to pixel_position
        float _dpx = static_cast<float>(0), _dpy = static_cast<float>(0);

        if (within_bounds_2d(top_y_index, left_x_index, height, width))
        {

          _dpx -= top_left * (1 - y_lerp) * _dg;
          _dpy -= top_left * (1 - x_lerp) * _dg;
        }
        if (within_bounds_2d(top_y_index, right_x_index, height, width))
        {
          _dpx += top_right * (1 - y_lerp) * _dg;
          _dpy -= top_right * (x_lerp)*_dg;
        }
        if (within_bounds_2d(bottom_y_index, left_x_index, height, width))
        {
          _dpx -= bottom_left * (y_lerp)*_dg;
          _dpy += bottom_left * (1 - x_lerp) * _dg;
        }
        if (within_bounds_2d(bottom_y_index, right_x_index, height, width))
        {
          _dpx += bottom_right * (y_lerp)*_dg;
          _dpy += bottom_right * (x_lerp)*_dg;
        }
        _dpx = _dpx * (width - 1.f) / 2;
        _dpy = _dpy * (height - 1.f) / 2;

        atomicAdd(dp + (((batch * NumSample + i) * height + y) * width + x) * 2 + 0,
                  static_cast<float>(_dpx));
        atomicAdd(dp + (((batch * NumSample + i) * height + y) * width + x) * 2 + 1,
                  static_cast<float>(_dpy));
      }
    }
  }
}

template <typename T>
__global__ void map_backward_kernel_w(const T *dout, const T *weight, const T *g, T *dw,
                                         T *pixel_position, int num, int chn, int height, int width, int NumSample)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < height + width - 1)
  {

    for (int batch = 0; batch < num; ++batch)
    {
      for (int plane = 0; plane < chn; ++plane)
      {
        float _dout = dout[(batch * chn + plane) * sp + y * width + x];

        if (z < width)
        {
          int i = z;
          float _g = g[(batch * chn + plane) * sp + y * width + i];
          dw[(batch * len + i) * sp + y * width + x] += _dout * _g;
        }
        else
        {
          int i = z - width;
          int j = i < y ? i : i + 1;

          float _g = g[(batch * chn + plane) * sp + j * width + x];

          dw[(batch * len + width + i) * sp + y * width + x] += _dout * _g;
        }
      }
    }
  }
}

template <typename T>
__global__ void map_backward_kernel_g(const T *dout, const T *weight, const T *g, T *dg,
                                         int num, int chn, int height, int width, int NumSample)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = NumSample;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn)
  {

    for (int batch = 0; batch < num; ++batch)
    {
      for (int i = 0; i < len; ++i)
      {
        float _dout = dout[(batch * chn + plane) * sp + y * width + i];
        float _w = weight[(batch * len + x) * sp + y * width + i];
        dg[(batch * chn + plane) * sp + y * width + x] += _dout * _w;
      }
    }
  }
}

/*
 * Implementations
 */
int _forward_cuda(int N, int C, int H, int W, int NumSample, at::Tensor t, at::Tensor f, at::Tensor weight, at::Tensor pixel_position)
{
  //Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;
  int d3 = NumSample;
  dim3 blocks(d1, d2, d3);
  //printf("%d %d %d %d %d \n",N,C,H,W,NumSample);
  forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(t.data<float>(), f.data<float>(), weight.data<float>(), pixel_position.data<float>(), N, C, H, W, NumSample);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

int _backward_cuda(int N, int C, int H, int W, int NumSample, const at::Tensor dw, const at::Tensor t, const at::Tensor f, const at::Tensor pixel_position, at::Tensor dt, at::Tensor df, at::Tensor dp)
{
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;
  int d3 = C;
  dim3 blocks(d1, d2, d3);
  // printf("%f\n", dw[0]);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  backward_kernel_all<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dw.data<float>(), t.data<float>(), f.data<float>(), pixel_position.data<float>(), dt.data<float>(), df.data<float>(), dp.data<float>(), N, C, H, W, NumSample);

  // Check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

int _map_forward_cuda(int N, int C, int H, int W, int NumSample, const at::Tensor weight, const at::Tensor g, const at::Tensor pixel_position, at::Tensor out)
{
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;
  int d3 = C;
  dim3 blocks(d1, d2, d3);
  map_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(weight.data<float>(), g.data<float>(), pixel_position.data<float>(), out.data<float>(), N, C, H, W, NumSample);

  // Check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

int _map_backward_cuda(int N, int C, int H, int W, int NumSample, const at::Tensor dout, const at::Tensor weight, const at::Tensor g, const at::Tensor pixel_position, at::Tensor dw, at::Tensor dg, at::Tensor dp)
{
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W + threads.x - 1) / threads.x;
  int d2 = (H + threads.y - 1) / threads.y;
  int d3 = NumSample;
  dim3 blocks(d1, d2, d3);
  map_backward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dout.data<float>(), weight.data<float>(), g.data<float>(), pixel_position.data<float>(), dw.data<float>(), dg.data<float>(), dp.data<float>(), N, C, H, W, NumSample);

  // Check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}
