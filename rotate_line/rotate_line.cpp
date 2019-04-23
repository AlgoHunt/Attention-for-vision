// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <torch/torch.h>
#include <vector>
// Forward definition of implementation functions



int _forward_cuda(int N, int C, int H, int W, int NumSample, at::Tensor t, at::Tensor f, at::Tensor weight, at::Tensor pixel_position);
int _backward_cuda(int N, int C, int H, int W,int NumSample, const at::Tensor dw, const at::Tensor t, const at::Tensor f,const at::Tensor pixel_position, at::Tensor dt, at::Tensor df, at::Tensor dp);
int _map_forward_cuda(int N, int C, int H, int W,int NumSample, const at::Tensor weight, const at::Tensor g, const at::Tensor pixel_position, at::Tensor out );
int _map_backward_cuda(int N, int C, int H, int W,int NumSample,const at::Tensor dout, const at::Tensor weight, const at::Tensor g, const at::Tensor pixel_position, at::Tensor dw, at::Tensor dg,at::Tensor dp);



// pixel_position : N, H, W, NumSample, 2
// weight: N, NumSample,H, W,
int forward_cuda(at::Tensor t, at::Tensor f, at::Tensor weight,at::Tensor pixel_position){
  int N, C, H, W, NumSample;
  N = t.size(0);
  C = t.size(1);
  H = t.size(2);
  W = t.size(3);
  NumSample = pixel_position.size(1);
  return _forward_cuda(N, C, H, W ,NumSample, t, f,  weight, pixel_position);
}

int backward_cuda(at::Tensor dw, at::Tensor t, at::Tensor f, at::Tensor pixel_position, at::Tensor dt,at::Tensor df ,at::Tensor dp) {

  int N, C, H, W, NumSample;
  N = t.size(0);
  C = t.size(1);
  H = t.size(2);
  W = t.size(3);
  NumSample = pixel_position.size(1);


  return _backward_cuda(N, C, H, W,NumSample, dw, t, f,pixel_position, dt, df , dp);
}

int map_forward_cuda(const at::Tensor weight, const at::Tensor g, const at::Tensor pixel_position, at::Tensor out) {

  int N, C, H, W, NumSample;
  N = g.size(0);
  C = g.size(1);
  H = g.size(2);
  W = g.size(3);
  NumSample = pixel_position.size(1);


  return _map_forward_cuda(N, C, H, W, NumSample, weight, g, pixel_position, out);
}


int map_backward_cuda(const at::Tensor dout, at::Tensor weight, const at::Tensor g, const at::Tensor pixel_position, at::Tensor dw,at::Tensor dg,at::Tensor dp) {

  int N, C, H, W, NumSample;
  N = g.size(0);
  C = g.size(1);
  H = g.size(2);
  W = g.size(3);
  NumSample = pixel_position.size(1);


  return _map_backward_cuda(N, C, H, W, NumSample, dout, weight, g, pixel_position, dw, dg, dp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward_cuda", &forward_cuda, "compute attention map with key and query");
  m.def("backward_cuda", &backward_cuda, "backward to key and query");
  m.def("map_forward_cuda", &map_forward_cuda, "compute output with value and attention map");
  m.def("map_backward_cuda", &map_backward_cuda, "backward to value and attention map");
}