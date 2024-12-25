import torch
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_value=1.0):
        ctx.lambda_value = lambda_value
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_value = ctx.lambda_value
        grad_input = grad_output.neg() * lambda_value
        return grad_input, None

def gradient_reversal(x, lambda_value=1.0):
    return GradientReversalLayer.apply(x, lambda_value)
