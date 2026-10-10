// test_inspect.cpp
#include <torch/extension.h>
#include <iostream>

void inspect(torch::Tensor a) {
    if (!a.defined()) {
        std::cout << "Tensor is undefined\n";
        return;
    }
    std::cout << "=== Tensor 基本信息 ===\n";
    std::cout << "dtype:          " << a.dtype() << "\n";
    std::cout << "device:         " << a.device() << "\n";
    std::cout << "layout:         " << a.layout() << "\n";
    std::cout << "sizes:          " << a.sizes() << "\n";
    std::cout << "strides:        " << a.strides() << "\n";
    std::cout << "dim:            " << a.dim() << "\n";
    std::cout << "numel:          " << a.numel() << "\n";
    std::cout << "element_size:   " << a.element_size() << " bytes\n";
    std::cout << "nbytes:         " << a.nbytes() << " bytes\n";
    std::cout << "storage_offset: " << a.storage_offset() << "\n";
    std::cout << "is_contiguous:  " << a.is_contiguous() << "\n";
    std::cout << "is_cuda:        " << a.is_cuda() << "\n";
    std::cout << "requires_grad:  " << a.requires_grad() << "\n";
    std::cout << "is_leaf:        " << a.is_leaf() << "\n";
    std::cout << "data_ptr:       " << a.data_ptr() << "\n";
    std::cout << "storage size:   " << a.storage().nbytes() << " bytes\n";
    std::cout << "use_count:      " << a.use_count() << "\n";
}
