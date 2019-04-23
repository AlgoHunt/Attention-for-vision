import math
import socket
import os
import random
from os import path
import torch
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
from torch import nn
from torch.nn import functional as F

_src_path = path.dirname(path.abspath(__file__))

if not socket.gethostname().startswith('msra'):
    folder = 'torch' + str(random.random())[2:]
    build_directory = os.path.join('/tmp', folder)
    os.mkdir(build_directory)
else:
    build_directory = None

try:
    import incenter_build as _ext
except ImportError:
    _ext = load(name="incenter_match_build",
                extra_cflags=["-O3"],
                sources=[path.join(_src_path, f) for f in [
                    "vanilla.cpp",
                    "vanilla.cu",
                ]],
                extra_cuda_cflags=["--expt-extended-lambda"],
                build_directory=build_directory)


def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


# pixel_position : N, S, H, W, 2
# weight : N,S,H,W
class Vanilla_Weight(autograd.Function):

    @staticmethod
    def forward(ctx, t, f):
        # Save context
        n, c, h, w = t.size()
        sample_num = (h)*(w)

        size = (n, sample_num, h, w)
        weight = torch.zeros(size, dtype=t.dtype,
                             layout=t.layout, device=t.device)

        _check_contiguous(weight)
        ret = _ext.forward_cuda(t, f, weight)
        if ret == 0:
            print('ra forward fail')
        # Output
        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        _check_contiguous(dt, df)
        ret = _ext.backward_cuda(dw.contiguous(), t, f, dt, df)
        if ret == 0:
            print('ra backward fail')
        _check_contiguous(dt, df)

        return dt, df, None


class Vanilla_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        # Save context
        out = torch.zeros_like(g)
        ret = _ext.map_forward_cuda(weight, g, out)
        if ret == 0:
            print('ra map forward fail')
        # Output
        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)
        ret = _ext.map_backward_cuda(dout.contiguous(), weight, g, dw, dg)
        if ret == 0:
            print('ra map  backward fail')
        _check_contiguous(dw, dg)

        return dw, dg, None


vanilla_weight = Vanilla_Weight.apply
vanilla_map = Vanilla_Map.apply


def is_same(tensorA, tensorB):
    if torch.all(torch.abs(tensorA-tensorB) < 1e-8):
        return True
    else:
        return False


def test_vanilla_weight():
    n, c, h, w = 1, 8, 25, 25
    key = torch.rand(n, c, h, w, requires_grad=True).float().cuda()
    query = torch.rand(n, c, h, w, requires_grad=True).float().cuda()

    weight_cuda = vanilla_weight(query, key)

    key_py = key.view(n, c, h * w).permute(0, 2, 1)

    query_py = query.view(n, c, h*w)

    weight_py = torch.matmul(key_py, query_py).view(n, h*w, h, w)

    if is_same(weight_py, weight_cuda):
        print('forward success')

    grad_terbute = torch.rand(weight_cuda.size()).cuda()
    weight_cuda *= grad_terbute
    weight_py *= grad_terbute

    grad_key_cuda, grad_query_cuda = torch.autograd.grad(
        weight_cuda.mean(), [key, query], retain_graph=True)
    grad_key_py, grad_query_py = torch.autograd.grad(
        weight_py.mean(), [key, query], retain_graph=True)

    if is_same(grad_key_cuda, grad_key_py):
        print('key grad correct')

    if is_same(grad_query_cuda, grad_query_py):
        print('query grad correct')


def test_vanilla_map():

    n, c, h, w = 1, 8, 25, 25
    value = torch.rand(n, c, h, w, requires_grad=True).cuda()
    weight = torch.rand(n, h*w, h, w, requires_grad=True).cuda()
    out_cuda = vanilla_map(weight, value)

    weight_py = weight.permute(0, 2, 3, 1)
    value_py = value.view(n, c, h*w).permute(0, 2, 1)
    out_py = torch.matmul(weight_py, value_py).squeeze(-2).permute(0, 3, 1, 2)

    if is_same(out_py, out_cuda):
        print('map forward success')

    grad_value_cuda, grad_weight_cuda = torch.autograd.grad(
        out_cuda.mean(), [value, weight], retain_graph=True)
    grad_value_py, grad_weight_py = torch.autograd.grad(
        out_py.mean(), [value, weight], retain_graph=True)

    print(grad_value_cuda-grad_value_py)
    if is_same(grad_value_cuda, grad_value_py):
        print('value grad correct')

    if is_same(grad_weight_cuda, grad_weight_py):
        print('weight grad correct')


if __name__ == '__main__':
    test_vanilla_weight()
    test_vanilla_map()
