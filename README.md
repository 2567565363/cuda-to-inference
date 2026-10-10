# CUDA to Inference

**Building a lightweight LLM inference engine from the ground up with CUDA and PyTorch extensions.**

`cuda-to-inference` is a hands-on project focused on GPU kernel development, CUDA optimization, and the low-level foundations of Transformer inference.

The goal is to build an open-source CUDA operator library first, then use it as the foundation for a lightweight, vLLM-like inference framework. Rather than relying entirely on existing operator libraries, this project explores how GPU kernels, tensor operations, neural network layers, and inference infrastructure fit together.

> **Status:** Work in progress. The current implementation focuses on custom CUDA operators and their PyTorch extension interfaces. It is not yet a complete LLM inference engine.

## Project Goals

- **Implement CUDA operators:** Build GPU kernels for fundamental tensor operations used in Transformer models.
- **Optimize GPU performance:** Explore memory coalescing, thread and block organization, parallel reduction, shared memory, and Tensor Core acceleration.
- **Integrate with PyTorch:** Expose custom CUDA kernels through a C++/CUDA extension with a Python interface.
- **Build neural network layers:** Compose low-level operators into reusable components such as `Linear` and `RMSNorm`.
- **Develop an inference engine:** Gradually build the components required to execute Transformer models and eventually support LLM inference.

## Implemented Operators

The project is being developed incrementally. The following table describes the current implementation scope.

| Operator | Status | Notes |
|---|---|---|
| Elementwise Add | Implemented / under development | Supports CUDA tensors; broadcasting support is being developed |
| Elementwise Multiply | Implemented | Custom CUDA kernel |
| GEMM | Implemented | Basic CUDA Core implementation; FP32 only |
| RMSNorm | Implemented | Custom CUDA kernel |
| Softmax | Implemented | Custom CUDA kernel |
| RoPE | Implemented | Rotary positional embedding |
| Linear | Under development | Python layer built on GEMM and bias addition |
| Tensor Core GEMM | Planned / under optimization | Targeting NVIDIA GPUs with `sm_89` support |
| BF16 GEMM | Planned | Required for the intended BF16 inference path |

Implementation status may change as the project evolves. Passing correctness tests does not necessarily imply performance parity with PyTorch, cuBLAS, or other optimized libraries.

## Architecture

The project separates the Python-facing API, the C++ extension interface, CUDA launch logic, and device-side computation.

```text
Python model / user code
          |
          v
    inference.layers
  (Linear, RMSNorm, ...)
          |
          v
     inference.ops
   (Python operator API)
          |
          v
      bindings.cpp
 (Tensor checks, allocation,
   Python/C++ bindings)
          |
          v
      CUDA launcher
 (dispatch, pointers, launch)
          |
          v
     CUDA kernels
 (GEMM, Add, RMSNorm, ...)
          |
          v
       NVIDIA GPU
```

### Design Principles

- **Clear interfaces:** Keep the Python API, C++ wrappers, and CUDA implementations separate.
- **Explicit tensor validation:** Check tensor properties such as device, dtype, shape, and memory layout at the extension boundary.
- **Incremental optimization:** Start with a correct baseline kernel before introducing more complex optimizations.
- **Numerical correctness:** Compare custom operator outputs against PyTorch reference implementations.
- **Reusable components:** Build neural network layers from the custom operators rather than treating each kernel as an isolated experiment.

## Repository Structure

```text
cuda-to-inference/
├── README.md
└── cudakernel/
    ├── setup.py
    ├── bindings.cpp
    ├── add.cu
    ├── mul.cu
    ├── gemm.cu
    ├── rmsnorm.cu
    ├── softmax.cu
    ├── rope.cu
    └── inference/
        ├── __init__.py
        ├── ops.py
        ├── layers.py
        ├── test.py
        └── _C.*.so
```

The compiled extension file (`_C.*.so`) is generated during the build and should not be treated as a portable, prebuilt binary.

## Requirements

The development environment currently targets:

- Python 3.10
- PyTorch with CUDA support
- NVIDIA GPU
- CUDA Toolkit and a compatible C++ compiler
- Linux development environment

The primary development GPU is an NVIDIA RTX 4090, targeting the `sm_89` architecture.

The exact CUDA Toolkit version, PyTorch CUDA build, and compiler must be compatible with the local environment. The development setup has used PyTorch built with CUDA 12.1 alongside a local CUDA 12.4 toolkit; compatibility warnings may need to be addressed depending on the installed toolchain.

## Build

Clone the repository and enter the CUDA project directory:

```bash
git clone https://github.com/2567565363/cuda-to-inference.git
cd cuda-to-inference/cudakernel
```

Activate your Python environment and install a compatible PyTorch build if needed. Then compile the extension:

```bash
python setup.py build_ext --inplace
```

This builds the C++/CUDA extension and places the compiled module in the Python package directory.

If importing the extension fails because the dynamic linker cannot locate PyTorch's shared libraries, check your environment's library paths and ensure the PyTorch libraries are discoverable at runtime.

## Run Tests

From the `cudakernel/` directory, run:

```bash
python -m inference.test
```

The existing test suite covers the current operators, including Add, Multiply, GEMM, RMSNorm, Softmax, and RoPE.

For individual operator tests, add test cases to the existing test suite or create separate test modules as the project grows. Tests should compare custom CUDA results with PyTorch reference implementations using tolerances appropriate to each dtype and operation.

## Example: Custom Linear Layer

The project is developing a PyTorch-style `Linear` layer on top of its custom GEMM and Add operators.

The current weight layout is `[in_features, out_features]`, matching the matrix multiplication convention:

\[
Y = XW + b
\]

A usage example is:

```python
import torch
from inference.layers import Linear

device = "cuda"
layer = Linear(
    in_features=256,
    out_features=128,
    bias=True,
    device=device,
    dtype=torch.float32,
)
x = torch.randn(512, 256, device=device, dtype=torch.float32)
y = layer(x)
print(y.shape)
```

Expected output shape:

```text
torch.Size([512, 128])
```

This example assumes that the Linear path, including GEMM and bias broadcasting, has been implemented and built successfully. The current GEMM implementation supports FP32 only. Weight and bias initialization should also be added to the layer implementation before using it as a general-purpose neural network component.

## Development Roadmap

### Phase 1 — Core CUDA Operators

- [x] Elementwise Add
- [x] Elementwise Multiply
- [x] Baseline FP32 GEMM
- [x] RMSNorm
- [x] Softmax
- [x] RoPE
- [ ] Complete broadcasting and edge-case coverage
- [ ] Expand correctness tests across shapes and dtypes

### Phase 2 — Kernel Optimization

- [x] Warp-level reduction experiments
- [x] Float4 vectorized memory access experiments
- [x] CUDA transpose optimization experiments
- [ ] Further GEMM tiling and memory-access optimization
- [ ] Tensor Core GEMM for supported data types
- [ ] Benchmark against PyTorch and cuBLAS
- [ ] Profile bottlenecks with NVIDIA Nsight tools

### Phase 3 — Neural Network Layers

- [x] RMSNorm layer interface
- [ ] Complete Linear with initialization and bias broadcasting
- [ ] Integrate optimized GEMM into Linear
- [ ] Add activation functions such as SiLU and GELU
- [ ] Improve dtype handling and numerical validation

### Phase 4 — Transformer Inference

- [ ] Implement the operator set required by a target Transformer architecture
- [ ] Build model components from the custom layers
- [ ] Load pretrained model weights
- [ ] Implement a minimal end-to-end inference path
- [ ] Explore KV cache management
- [ ] Explore continuous batching and memory-efficient serving

The initial model target is a Qwen-family Transformer. The exact supported model and precision will depend on the completion of the required operators and inference components.

## Performance and Correctness

Correctness and performance are evaluated separately.

- **Correctness:** Compare custom kernels against PyTorch reference operations.
- **Performance:** Measure kernel latency and effective memory bandwidth, where applicable.
- **Profiling:** Use NVIDIA Nsight tools to investigate occupancy, memory traffic, register pressure, and instruction execution.
- **Reproducibility:** Record input shapes, dtype, GPU model, compiler flags, and benchmark configuration when reporting results.

Performance claims will be added alongside reproducible benchmark configurations as the relevant kernels stabilize.

## Scope

This project is primarily a learning and engineering effort focused on understanding the low-level mechanisms behind GPU inference. It is not currently intended to replace mature production libraries such as cuBLAS or established inference engines.

The long-term objective is to understand and implement the path from a CUDA kernel to a functioning Transformer inference system.

## Contributing

Issues, bug reports, optimization ideas, and discussions are welcome. Contributions should include reproducible test cases where applicable, especially for numerical correctness and performance improvements.

## License

A license has not yet been specified. A license should be added to the repository before the project is distributed for reuse or external contributions.
