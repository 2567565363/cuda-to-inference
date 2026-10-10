import unittest
import torch
import inference
class TestInferenceOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        cls.ops = inference._C
        cls.device = torch.device("cuda")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Extension: {inference._C.__file__}")
    def make_tensor(self, shape, dtype):
        return torch.randn(shape, device=self.device, dtype=dtype)
    def check_output(self, actual, expected, rtol=None, atol=None):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(torch.isfinite(actual).all().item(), "output contains NaN or Inf")
        if rtol is None:
            rtol = 2e-2 if actual.dtype == torch.bfloat16 else 1e-4
        if atol is None:
            atol = 2e-2 if actual.dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    def test_add(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                a = self.make_tensor((1024,), dtype)
                b = self.make_tensor((1024,), dtype)
                out = self.ops.add(a, b)
                ref = a + b
                self.check_output(out, ref)
        print("add PASS")
    def test_mul(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                a = self.make_tensor((3, 1024), dtype)
                b = self.make_tensor((3, 1024), dtype)
                out = self.ops.mul(a, b)
                ref = a * b
                self.check_output(out, ref)
        print("mul PASS")
    def test_rmsnorm(self):
        eps = 1e-6
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = self.make_tensor((2, 8, 4096), dtype)
                weight = self.make_tensor((4096,), dtype)
                out = self.ops.RMSNorm(x, weight, eps)
                x32 = x.float()
                w32 = weight.float()
                ref = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps) * w32
                ref = ref.to(dtype)
                self.check_output(out, ref)
        print("RMSNorm PASS")
    def test_rmsnorm_v2(self):
        eps = 1e-6
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = self.make_tensor((2, 8, 4096), dtype)
                weight = self.make_tensor((4096,), dtype)
                out = self.ops.RMSNorm_v2(x, weight, eps)
                x32 = x.float()
                w32 = weight.float()
                ref = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps) * w32
                ref = ref.to(dtype)
                self.check_output(out, ref)
        print("RMSNorm_v2 PASS")
    def test_softmax(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = self.make_tensor((2, 8, 1024), dtype)
                out = self.ops.softmax(x)
                ref = torch.softmax(x.float(), dim=-1).to(dtype)
                self.check_output(out, ref)
                sums = out.float().sum(dim=-1)
                torch.testing.assert_close(sums, torch.ones_like(sums), rtol=2e-2, atol=2e-2)
        print("softmax PASS")
    def test_rope(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                batch, seq, hidden = 2, 16, 128
                x = self.make_tensor((batch, seq, hidden), dtype)
                inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden, 2, device=self.device, dtype=torch.float32) / hidden))
                positions = torch.arange(seq, device=self.device, dtype=torch.float32)
                angles = positions[:, None] * inv_freq[None, :]
                angles = torch.cat((angles, angles), dim=-1).unsqueeze(0)
                cos = angles.cos().expand(1, seq, hidden).contiguous().to(dtype)
                sin = angles.sin().expand(1, seq, hidden).contiguous().to(dtype)
                out = self.ops.rope(x, cos, sin)
                x32 = x.float()
                cos32 = cos.float()
                sin32 = sin.float()
                half = hidden // 2
                rotate_half = torch.cat((-x32[..., half:], x32[..., :half]), dim=-1)
                ref = (x32 * cos32 + rotate_half * sin32).to(dtype)
                self.check_output(out, ref, rtol=3e-2 if dtype == torch.bfloat16 else 1e-4, atol=3e-2 if dtype == torch.bfloat16 else 1e-5)
        print("rope PASS")
    def test_gemm(self):
        shapes = [
            (2, 32, 32, 32),
            (4, 64, 128, 96),
            (8, 37, 65, 53),
            (2000, 32, 64, 48),
        ]
        for batch, M, K, N in shapes:
            with self.subTest(batch=batch, M=M, K=K, N=N):
                a = self.make_tensor((batch, M, K), torch.float32)
                b = self.make_tensor((batch, K, N), torch.float32)
                out = self.ops.gemm(a, b)
                ref = torch.bmm(a, b)
                self.check_output(out, ref, rtol=1e-4, atol=1e-4)
        print("gemm PASS")
if __name__ == "__main__":
    print(f"Package: {inference.__file__}")
    unittest.main(verbosity=2)