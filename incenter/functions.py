#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by: algohunt
# Microsoft Research Asia & Peking University 
# lilingzhi@pku.edu.cn
# Copyright (c) 2019

import os
from os import path
import torch
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
from torch import nn
from torch.nn import functional as F

_src_path = path.dirname(path.abspath(__file__))
_ext = load(name="incenter_build",
                extra_cflags=["-O3"],
                sources=[path.join(_src_path, f) for f in [
                    "ia.cpp",
                    "ia.cu",
                ]],
                extra_cuda_cflags=["--expt-extended-lambda  -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__"]
            )



def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class IA_Weight(autograd.Function):

    @staticmethod
    def forward(ctx, t, f, factor):
        
        n, c, h, w = t.size()
        sample_num = (h//factor)*(w//factor)

        size = (n, sample_num, h, w)
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)

        _check_contiguous(weight)
        ret = _ext.ia_forward_cuda(t, f, weight, factor)
        if ret == 0:
            print('incenter attention forward fail')
       
        ctx.save_for_backward(t, f)
        ctx.factor = factor

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        _check_contiguous(dt, df)
        ret = _ext.ia_backward_cuda(dw.contiguous(), t, f, ctx.factor, dt, df)
        if ret == 0:
            print('incenter attention backward fail')
        _check_contiguous(dt, df)

        return dt, df, None

class IA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g, factor):
        
        out = torch.zeros_like(g)
        ret = _ext.ia_map_forward_cuda(weight, g, factor, out)
        if ret == 0:
            print('incenter attention map forward fail')
       
        ctx.save_for_backward(weight, g)
        ctx.factor=factor

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors
        factor = ctx.factor

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)
        ret = _ext.ia_map_backward_cuda(dout.contiguous(), weight, g, factor, dw, dg)
        if ret == 0:
            print('incenter attention map backward fail')
        _check_contiguous(dw, dg)

        return dw, dg, None


ia_weight = IA_Weight.apply
ia_map = IA_Map.apply


import math

def is_same(tensorA,tensorB):
    if torch.all(torch.abs(tensorA-tensorB)<1e-6):
        return True
    else:
        return False


def test_ia_weight():
    key = torch.rand(1, 8, 97, 97, requires_grad=True).float().cuda()
    query = torch.rand(1, 8, 97, 97, requires_grad=True).float().cuda()
    n, c, h, w = key.size()
    factor = 7
  
    hnum = h // factor
    wnum = w // factor
    hpad = int(math.ceil(h / factor) * factor - h)
    wpad = int(math.ceil(w / factor) * factor - w)

    weight_cuda = ia_weight(query, key, factor)

    key_py = F.pad(key, (wnum//2 ,math.ceil( wnum/2)-1 , hnum//2, math.ceil(hnum/2)-1))
    key_py = F.unfold(key_py, (hnum, wnum), dilation=1, padding=0, stride=1)\
    .view(1, c, hnum * wnum, h, w).permute(0, 3, 4, 2, 1)

    query_py = query.permute(0, 2, 3, 1).unsqueeze(-1)
    weight_py = torch.matmul(key_py, query_py).squeeze(-1).permute(0, 3, 1, 2)

    if is_same(weight_py, weight_cuda):
        print('incenter attention forward success')

    grad_terbute = torch.rand(weight_cuda.size()).cuda()
    weight_cuda *= grad_terbute
    weight_py *= grad_terbute

    grad_key_cuda,grad_query_cuda = torch.autograd.grad(weight_cuda.mean(), [key,query], retain_graph=True)
    grad_key_py,grad_query_py = torch.autograd.grad(weight_py.mean(), [key,query], retain_graph=True)


    if is_same(grad_key_cuda, grad_key_py):
        print('key grad correct')

    if is_same(grad_query_cuda,grad_query_py):
        print('query grad correct')

def test_ia_map():



    n,c,h,w = 1,1,7,7
    factor = 2


    key = torch.rand(1, 1, 7, 7, requires_grad=True).float().cuda()
    query = torch.rand(1, 1, 7, 7, requires_grad=True).float().cuda()
    n, c, h, w = key.size()
    factor = 2

    hnum = h // factor
    wnum = w // factor

    pad = int((factor-1)/2)

    value = torch.rand(n, c, h, w, requires_grad=True).cuda()
    weight = torch.rand(n, hnum*wnum, h, w, requires_grad=True).cuda()
    out_cuda = ia_map(weight, value, factor)

    value_pad = F.pad(value, (wnum//2 , wnum//2 , hnum//2, hnum//2))
    value_unfold = F.unfold(value_pad, (hnum, wnum), dilation=1, padding=0, stride=1).view(n, c, hnum*wnum, h, w).permute(0, 3, 4, 2, 1)
    weight_v = weight.permute(0, 2, 3, 1).unsqueeze(-2)
    out_py = torch.matmul(weight_v, value_unfold).squeeze(-2).permute(0, 3, 1, 2)

    if is_same(out_py,out_cuda):
        print('incenter attention map forward success')

    grad_value_cuda,grad_weight_cuda = torch.autograd.grad(out_cuda.mean(), [value, weight], retain_graph=True)
    grad_value_py,grad_weight_py = torch.autograd.grad(out_py.mean(), [value, weight], retain_graph=True)

    if is_same(grad_value_cuda, grad_value_py):
        print('value grad correct')

    if is_same(grad_weight_cuda, grad_weight_py):
        print('weight grad correct')



if __name__ == '__main__':
    test_ia_weight()
    test_ia_map()





