import torch, inference

a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = inference.add(a, b)
print(torch.allclose(c, a + b))   # 应为 True

d = inference.mul(a, b)
print(torch.allclose(d, a * b))   # 应为 True