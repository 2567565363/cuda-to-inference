#include <torch/extension.h>
#include <vector>
#include <algorithm>
//先声明函数
void add_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor c);
void mul_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor c);
void RMSNorm_cuda(torch::Tensor a,torch::Tensor b ,torch::Tensor weight,float eps);
void RMSNorm_cuda_v2(torch::Tensor a,torch::Tensor b,torch::Tensor weight,float eps);
void softmax_cuda(torch::Tensor a,torch::Tensor b);
void rope_cuda(torch::Tensor in,torch::Tensor out,torch::Tensor cos,torch::Tensor sin);
void gemm_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor c);
//CPP端最后一次封装

std::vector<int64_t> infer_broadcast_shape(torch::IntArrayRef a,torch::IntArrayRef b)
{
    size_t ndim = std::max(a.size(),b.size());
    std::vector<int64_t> out(ndim,1);
    for (size_t i = 0; i < ndim; ++i)
    {
        int64_t da = i < ndim - a.size() ? 1 : a[i - (ndim - a.size())];
        int64_t db = i < ndim - b.size() ? 1 : b[i - (ndim - b.size())];
        TORCH_CHECK(da == db || da == 1 || db == 1,"shape is not broadcastable");
        out[ndim - 1 - i] = da == 1 ? db : da;
    }
    return out;
}
torch::Tensor add(torch::Tensor a,torch::Tensor b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),"a/b must be CUDA");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),"dtype mismatch");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),"must be contiguous");
    TORCH_CHECK(a.device() == b.device(),"device mismatch");
    auto shape = infer_broadcast_shape(a.sizes(),b.sizes());
    TORCH_CHECK(shape.size() <= 64,"too many dimensions");
    auto c = torch::empty(shape,a.options());
    add_cuda(a,b,c);
    return c;
}
torch::Tensor mul(torch::Tensor a,torch::Tensor b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "dtype mismatch");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    TORCH_CHECK(a.device() == b.device(), "device mismatch");

    auto c=torch::empty_like(a);
    mul_cuda(a,b,c);
    return c;
}

//y=x*weight/RMS(x)
torch::Tensor RMSNorm(torch::Tensor a,torch::Tensor weight,float eps=1e-6f)
{
    TORCH_CHECK(a.is_cuda(),"a must be CUDA");
    TORCH_CHECK(weight.is_cuda(),"weight must be CUDA");
    TORCH_CHECK(a.is_contiguous(),"a must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),"weight must be contiguous");
    TORCH_CHECK(a.dim() >= 2,"a must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 1,"weight must be 1D");
    TORCH_CHECK(a.size(-1) == weight.size(0),"hidden mismatch");//a的最后一个维度和weight的维度必须相同

    torch::Tensor b=torch::empty_like(a);
    RMSNorm_cuda(a,b,weight,eps);
    return b;
}

torch::Tensor RMSNorm_v2(torch::Tensor a,torch::Tensor weight,float eps=1e-6f)
{
    TORCH_CHECK(a.is_cuda(),"a must be CUDA");
    TORCH_CHECK(weight.is_cuda(),"weight must be CUDA");
    TORCH_CHECK(a.is_contiguous(),"a must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),"weight must be contiguous");
    TORCH_CHECK(a.dim() >= 2,"a must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 1,"weight must be 1D");
    TORCH_CHECK(a.size(-1) == weight.size(0),"hidden mismatch");//a的最后一个维度和weight的维度必须相同

    torch::Tensor b=torch::empty_like(a);
    RMSNorm_cuda_v2(a,b,weight,eps);
    return b;
}

torch::Tensor softmax(torch::Tensor a)
{
    TORCH_CHECK(a.is_cuda(),"a must be CUDA");
    TORCH_CHECK(a.is_contiguous(),"a must be contiguous");
    TORCH_CHECK(a.dim()>=2,"a must have at least 2 dimensions");

    torch::Tensor b=torch::empty_like(a);
    softmax_cuda(a,b);
    return b;
}

torch::Tensor rope(torch::Tensor a,torch::Tensor cos,torch::Tensor sin)
{
    TORCH_CHECK(a.is_cuda(),"a must be CUDA");
    TORCH_CHECK(cos.is_cuda(),"cos must be CUDA");
    TORCH_CHECK(sin.is_cuda(),"sin must be CUDA");
    TORCH_CHECK(a.is_contiguous(),"a must be contiguous");
    TORCH_CHECK(cos.is_contiguous(),"cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(),"sin must be contiguous");
    TORCH_CHECK(a.scalar_type()==cos.scalar_type(),"dtype mismatch");
    TORCH_CHECK(a.scalar_type()==sin.scalar_type(),"dtype mismatch");
    TORCH_CHECK(a.device()==cos.device(),"device mismatch");
    TORCH_CHECK(a.device()==sin.device(),"device mismatch");
    TORCH_CHECK(a.dim()>=3,"a must have at least 3 dimensions");
    TORCH_CHECK(cos.dim()==3,"cos must be 3D");
    TORCH_CHECK(sin.dim()==3,"sin must be 3D");
    TORCH_CHECK(a.size(-1)%2==0,"hidden_size must be even");
    TORCH_CHECK(cos.sizes()==sin.sizes(),"cos/sin shape mismatch");
    TORCH_CHECK(a.size(-2)==cos.size(-2),"sequence length mismatch");
    TORCH_CHECK(a.size(-3)==cos.size(-3) || cos.size(-3)==1,"batch dimension mismatch");
    TORCH_CHECK(a.size(-1)==cos.size(-1),"hidden dimension mismatch");
    torch::Tensor b=torch::empty_like(a);
    rope_cuda(a,b,cos,sin);
    return b;
}

torch::Tensor gemm(torch::Tensor a,torch::Tensor b)
{
    TORCH_CHECK(a.is_cuda(),"a must be CUDA");
    TORCH_CHECK(b.is_cuda(),"b must be CUDA");
    TORCH_CHECK(a.is_contiguous(),"a must be contiguous");
    TORCH_CHECK(b.is_contiguous(),"b must be contiguous");
    TORCH_CHECK(a.dim()==3,"a must be 3D");
    TORCH_CHECK(b.dim()==3,"b must be 3D");
    TORCH_CHECK(a.scalar_type()==b.scalar_type(),"dtype mismatch");
    TORCH_CHECK(a.device()==b.device(),"device mismatch");
    TORCH_CHECK(a.size(0)==b.size(0),"batch mismatch");
    TORCH_CHECK(a.size(2)==b.size(1),"matrix shape mismatch");
    torch::Tensor c=torch::empty({a.size(0),a.size(1),b.size(2)},a.options());
    gemm_cuda(a,b,c);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", &add, "CUDA add");//Python 中的函数名,C++中的函数地址，函数说明
    m.def("mul", &mul, "CUDA mul");
    m.def("RMSNorm",&RMSNorm,"CUDA RMSNorm");
    m.def("RMSNorm_v2",&RMSNorm_v2,"CUDA RMSNorm_v2");
    m.def("softmax",&softmax,"CUDA softmax");
    m.def("rope",&rope,"CUDA rope");
    m.def("gemm",&gemm,"CUDA Batched GEMM");
}