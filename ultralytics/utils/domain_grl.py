# ——*——code:UTF-8——*——
# Author : airy
# DATA : 2023/1/17 上午8:49
import torch


class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)


# import torch
# from torch.autograd import Function

# class GradientReversalLayer(Function):
#     @staticmethod
#     def forward(ctx, x, lambda_value=1.0):
#         ctx.lambda_value = lambda_value
#         return x.clone()

#     @staticmethod
#     def backward(ctx, grad_output):
#         lambda_value = ctx.lambda_value
#         grad_input = grad_output.neg() * lambda_value
#         return grad_input, None

# def gradient_reversal(x, lambda_value=1.0):
#     return GradientReversalLayer.apply(x, lambda_value)
