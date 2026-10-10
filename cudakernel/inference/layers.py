import torch
from torch import nn
from . import ops
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps
    def forward(self, x):
        return ops.rms_norm(x, self.weight, self.eps)

class Linear(nn.Module):
    """
    自定义 Linear 层。
    - weight 形状: [in_features, out_features]  (对齐 GEMM 的 x @ W)
    - bias   形状: [out_features]
    - 前向: y = ops.linear(x, weight) + bias
    """

    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        factory_kwargs = {"device": device, "dtype": dtype}

        # ---- 参数：按 [in, out] 存 ----
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)


    def forward(self, x):
        # ops.linear 内部就是 x @ weight，weight 已是 [in, out]
        y = ops.linear(x, self.weight)
        if self.bias is not None:
            y = ops.add(y, self.bias)
        return y
