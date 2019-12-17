// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <torch/torch.h>
#include <vector>
// Forward definition of implementation functions

int _forward_cuda(int N, int C, int H, int W, at::Tensor t, at::Tensor f, at::Tensor weight);
int _backward_cuda(int N, int C, int H, int W, const at::Tensor dw, const at::Tensor t, const at::Tensor f, at::Tensor dt, at::Tensor df);
int _map_forward_cuda(int N, int C, int H, int W, const at::Tensor weight, const at::Tensor g, at::Tensor out);
int _map_backward_cuda(int N, int C, int H, int W, const at::Tensor dout, const at::Tensor weight, const at::Tensor g, at::Tensor dw, at::Tensor dg);

// pixel_position : N, H, W, NumSample, 2
// weight: N, NumSample,H, W,
int forward_cuda(at::Tensor t, at::Tensor f, at::Tensor weight)
{
  int N, C, H, W;
  N = t.size(0);
  C = t.size(1);
  H = t.size(2);
  W = t.size(3);
  return _forward_cuda(N, C, H, W, t, f, weight);
}

int backward_cuda(at::Tensor dw, at::Tensor t, at::Tensor f, at::Tensor dt, at::Tensor df)
{

  int N, C, H, W;
  N = t.size(0);
  C = t.size(1);
  H = t.size(2);
  W = t.size(3);

  return _backward_cuda(N, C, H, W, dw, t, f, dt, df);
}

int map_forward_cuda(const at::Tensor weight, const at::Tensor g, at::Tensor out)
{

  int N, C, H, W;
  N = g.size(0);
  C = g.size(1);
  H = g.size(2);
  W = g.size(3);

  return _map_forward_cuda(N, C, H, W, weight, g, out);
}

int map_backward_cuda(const at::Tensor dout, at::Tensor weight, const at::Tensor g,  at::Tensor dw, at::Tensor dg)
{

  int N, C, H, W;
  N = g.size(0);
  C = g.size(1);
  H = g.size(2);
  W = g.size(3);

  return _map_backward_cuda(N, C, H, W, dout, weight, g, dw, dg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward_cuda", &forward_cuda, "compute attention map w.r.t key and query");
  m.def("backward_cuda", &backward_cuda, "backward to key and query");
  m.def("map_forward_cuda", &map_forward_cuda, "compute output w.r.t value and attention map");
  m.def("map_backward_cuda", &map_backward_cuda, "backward to value and attention map");
}