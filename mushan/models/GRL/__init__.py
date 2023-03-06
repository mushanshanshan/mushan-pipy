import torch

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGrad.apply(input_, self._alpha)


class RevGrad(torch.autograd.Function):
    """
    A gradient reversal layer.
    This layer has no parameters, and simply reverses the gradient in the backward pass.
    See https://www.codetd.com/en/article/11984164, https://github.com/janfreyberg/pytorch-revgrad
    """
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None
