from os import path
import torch
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
from torch import nn
from torch.nn import functional as F

_src_path = path.dirname(path.abspath(__file__))
_ext = load(name="rotate_line_attention",
            extra_cflags=["-O3"],
            sources=[path.join(_src_path, f) for f in [
                "rotate_line.cpp",
                "rotate_line.cu",
            ]],
            extra_cuda_cflags=["--expt-extended-lambda"])


class RotateGen(nn.Module):
    def __init__(self, n, c, h, w, sample_num):
        super(RotateGen, self).__init__()
        self.n = n
        self.c = c
        self.h = h
        self.w = w
        self.sample_num = sample_num
        affine_matrix = torch.tensor([[[1, 0, 0], [0, 1, 0]]]).float()
        raw_position = F.affine_grid(affine_matrix, torch.Size((1, c, h, w)))
        new_x = torch.linspace(-1, 1, sample_num).view(1,
                                                       sample_num, 1, 1).repeat(n, 1, h, w)
        raw_x = raw_position[:, :, :, 0].unsqueeze(0)
        raw_y = raw_position[:, :, :, 1].unsqueeze(0)
        self.new_x = nn.Parameter(new_x, False)
        self.raw_x = nn.Parameter(raw_x, False)
        self.raw_y = nn.Parameter(raw_y, False)

    def forward(self, slope):
        new_y = (self.new_x - self.raw_x) * slope + self.raw_y
        new_x = self.new_x.unsqueeze(-1)
        new_y = new_y.unsqueeze(-1)

        grid = torch.cat([new_x, new_y], dim=-1)
        return grid


class RotateSampler(nn.Module):
    def __init__(self, n, c, h, w, sample_num):
        super(RotateSampler, self).__init__()
        self.n = n
        self.c = c
        self.h = h
        self.w = w
        self.sample_num = sample_num
        affine_matrix = torch.tensor([[[1, 0, 0], [0, 1, 0]]]).float()
        raw_position = F.affine_grid(affine_matrix, torch.Size((1, c, h, w)))
        new_x = torch.linspace(-1, 1, sample_num).view(1,
                                                       sample_num, 1, 1).repeat(n, 1, h, w)
        raw_x = raw_position[:, :, :, 0].unsqueeze(0)
        raw_y = raw_position[:, :, :, 1].unsqueeze(0)
        new_z = torch.zeros_like(new_x).unsqueeze(-1)
        self.new_x = nn.Parameter(new_x, False)
        self.raw_x = nn.Parameter(raw_x, False)
        self.raw_y = nn.Parameter(raw_y, False)
        self.new_z = nn.Parameter(new_z, False)

    def forward(self, feature_map, slope):
        new_y = (self.new_x - self.raw_x) * slope + self.raw_y
        new_x = self.new_x.unsqueeze(-1)
        new_y = new_y.unsqueeze(-1)

        grid = torch.cat([new_x, new_y, self.new_z], dim=-1)
        feature_map = feature_map.unsqueeze(2)
        sample = F.grid_sample(feature_map, grid)
        return sample


def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")

# pixel_position : N, S, H, W, 2
# weight : N,S,H,W


class RL_Weight(autograd.Function):

    @staticmethod
    def forward(ctx, t, f, pixel_position):
        # Save context
        n, c, h, w = t.size()
        sample_num = pixel_position.size(1)
        size = (n, sample_num, h, w)
        weight = torch.zeros(size, dtype=t.dtype,
                             layout=t.layout, device=t.device)

        _check_contiguous(weight)
        ret = _ext.forward_cuda(t, f, weight, pixel_position)
        if ret == 0:
            print('rotate line attention forward fail')
        # Output
        ctx.save_for_backward(t, f, pixel_position)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f, pixel_position = ctx.saved_tensors

        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        dp = torch.zeros_like(pixel_position)
        _check_contiguous(dt, df, dp)
        ret = _ext.backward_cuda(
            dw.contiguous(), t, f, pixel_position, dt, df, dp)
        if ret == 0:
            print('rotate line attention backward fail')
        _check_contiguous(dt, df, dp)

        return dt, df, dp


class RL_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g, pixel_position):
        # Save context
        out = torch.zeros_like(g)
        ret = _ext.map_forward_cuda(weight, g, pixel_position, out)
        if ret == 0:
            print('rotate line attention map forward fail')
        # Output
        ctx.save_for_backward(weight, g, pixel_position)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g, pixel_position = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)
        dp = torch.zeros_like(pixel_position)
        ret = _ext.map_backward_cuda(
            dout.contiguous(), weight, g, pixel_position, dw, dg, dp)
        if ret == 0:
            print('rotate line attention backward fail')
        _check_contiguous(dw, dg, dp)

        return dw, dg, dp


rl_weight = RL_Weight.apply
rl_map = RL_Map.apply


def is_same(tensorA, tensorB):
    if torch.all(torch.abs(tensorA-tensorB) < 1e-4):
        return True
    else:
        return False


def test_rl_weight():
    n, c, h, w, sample_num = 1, 1, 5, 5, 3
    key = torch.randn((n, c, h, w), requires_grad=True).float().cuda()
    query = torch.randn((n, c, h, w), requires_grad=True).float().cuda()
    slope = torch.randn((n, 1, h, w), requires_grad=True).float().cuda()
    rg = RotateGen(n, c, h, w, sample_num)
    rg.cuda()
    position_grid = rg(slope)

    # cuda forward
    weight = rl_weight(query, key, position_grid)

    # cuda backward
    grad_key_cuda, grad_query_cuda, grad_position_cuda = \
        torch.autograd.grad(
            weight.sum(), [key, query, position_grid], retain_graph=True)

    # python forward
    key_ = key.unsqueeze(2)
    new_z = torch.zeros(n, sample_num, h, w, 1).cuda()
    grid = torch.cat([position_grid, new_z], dim=-1)
    key_ = F.grid_sample(key_, grid, padding_mode="zeros")
    key_ = key_.permute(0, 3, 4, 2, 1)
    query_ = query.permute(0, 2, 3, 1).unsqueeze(-1)
    sim_map = torch.matmul(key_, query_).squeeze(-1).permute(0, 3, 1, 2)

    # python backward
    grad_key_py, grad_query_py, grad_position_py = \
        torch.autograd.grad(
            sim_map.sum(), [key, query, position_grid], retain_graph=True)

    if is_same(sim_map, weight):
        print('rotate line attention weight forward correct')
    if is_same(grad_key_cuda, grad_key_py):
        print('key grad correct')
    if is_same(grad_query_cuda, grad_query_py):
        print('query grad correct')
    if is_same(grad_position_cuda, grad_position_py):
        print('position grad correct')


def test_rl_map_forward():
    n, c, h, w, sample_num = 1, 1, 5, 5, 3

    value = torch.randn((n, c, h, w), requires_grad=True).float().cuda()
    slope = torch.randn((n, 1, h, w), requires_grad=True).float().cuda()
    weight = torch.randn((n, sample_num, h, w), requires_grad=True).cuda()

    rg = RotateGen(n, c, h, w, sample_num)
    rg.cuda()
    position_grid = rg(slope)

    # python forward
    value_ = value.unsqueeze(2)
    new_z = torch.zeros(n, sample_num, h, w, 1).cuda()
    grid = torch.cat([position_grid, new_z], dim=-1)
    value_ = F.grid_sample(value_, grid, padding_mode="zeros")
    value_ = value_.permute(0, 3, 4, 1, 2)
    weight_ = weight.permute(0, 2, 3, 1).unsqueeze(-1)
    context = torch.matmul(value_, weight_).squeeze(-2)
    context1 = context.permute(0, 3, 1, 2).contiguous()

    # cuda forward
    context2 = rl_map(weight, value, position_grid)

    grad_weight_py, grad_value_py, grad_position_py = \
        torch.autograd.grad(
            context1.sum(), [weight, value, position_grid], retain_graph=True)
    grad_weight_cuda, grad_value_cuda, grad_position_cuda = \
        torch.autograd.grad(
            context2.sum(), [weight, value, position_grid], retain_graph=True)

    if is_same(context2, context1):
        print('rotate line attention map forward correct')
    if is_same(grad_weight_cuda, grad_weight_py):
        print('weight grad correct')
    if is_same(grad_value_cuda, grad_value_py):
        print('value grad correct')
    if is_same(grad_position_cuda, grad_position_py):
        print('position grad correct')


if __name__ == '__main__':
    test_rl_weight()
    test_rl_map_forward()
