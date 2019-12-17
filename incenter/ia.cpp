// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <torch/extension.h>
#include <vector>
// Forward definition of implementation functions

int _ia_forward_cuda(int N, int C, int H, int W, int factor, at::Tensor t, at::Tensor f, at::Tensor weight);
int _ia_backward_cuda(int N, int C, int H, int W, int factor, const at::Tensor dw, const at::Tensor t, const at::Tensor f, at::Tensor dt, at::Tensor df);
int _ia_map_forward_cuda(int N, int C, int H, int W, int factor, const at::Tensor weight, const at::Tensor g, at::Tensor out);
int _ia_map_backward_cuda(int N, int C, int H, int W, int factor, const at::Tensor dout, const at::Tensor weight, const at::Tensor g, at::Tensor dw, at::Tensor dg);

// pixel_position : N, H, W, NumSample, 2
// weight: N, NumSample,H, W,
int ia_forward_cuda(at::Tensor t, at::Tensor f, at::Tensor weight, int factor)
{
  int N, C, H, W;
  N = t.size(0);
  C = t.size(1);
  H = t.size(2);
  W = t.size(3);
  return _ia_forward_cuda(N, C, H, W, factor, t, f, weight);
}

int ia_backward_cuda(at::Tensor dw, at::Tensor t, at::Tensor f, int factor, at::Tensor dt, at::Tensor df)
{

  int N, C, H, W;
  N = t.size(0);
  C = t.size(1);
  H = t.size(2);
  W = t.size(3);

  return _ia_backward_cuda(N, C, H, W, factor, dw, t, f, dt, df);
}

int ia_map_forward_cuda(const at::Tensor weight, const at::Tensor g, const int factor, at::Tensor out)
{

  int N, C, H, W;
  N = g.size(0);
  C = g.size(1);
  H = g.size(2);
  W = g.size(3);

  return _ia_map_forward_cuda(N, C, H, W, factor, weight, g, out);
}

int ia_map_backward_cuda(const at::Tensor dout, at::Tensor weight, const at::Tensor g, const int factor, at::Tensor dw, at::Tensor dg)
{

  int N, C, H, W;
  N = g.size(0);
  C = g.size(1);
  H = g.size(2);
  W = g.size(3);

  return _ia_map_backward_cuda(N, C, H, W, factor, dout, weight, g, dw, dg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("ia_forward_cuda", &ia_forward_cuda,  "compute attention map w.r.t key and query");
  m.def("ia_backward_cuda", &ia_backward_cuda, "backward to key and query");
  m.def("ia_map_forward_cuda", &ia_map_forward_cuda, "compute output w.r.t value and attention map");
  m.def("ia_map_backward_cuda", &ia_map_backward_cuda, "backward to value and attention map");
}
