#include "../headers/functors.cuh"
#include "../headers/matrix.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/types"
#include <cuda_device_runtime_api.h>
#include <iostream>

using FloatT = float32;

int main(int argc, char const* argv[])
{
    // clang-format off
    FloatT wv[] = {0.254053,  -0.247817, -0.427396,
                   -0.014449,  0.147829,   0.929250,
                   -0.807114,  -0.068262,  0.218884, 
                   -0.453263,  0.800626,  -0.321338};

    FloatT bv[] = {-0.118935, -0.542531, 0.981391, -1.293095};
    // clang-format off

    Matrix<FloatT> x(3 , 1); 
    Matrix<FloatT> target(4, 1);
    fillCPU(x, 1.f);
    fillCPU(target, 1.f);

    Linear<FloatT, Sigmoid<FloatT>> l(3, 4, wv, bv);
    MSE<FloatT> mse(4, 1, true); 

    const auto& y = l.forward(x);
    cudaErrCheck(cudaDeviceSynchronize());
    const auto& err = mse.forward(y, target);
    cudaErrCheck(cudaDeviceSynchronize());

    std::cout << "y: " << y << std::endl;
    std::cout << "err_vec: " << mse.output_vec << std::endl;
    std::cout << "err: " << err << std::endl;

    mse.backward(err, target);

    //std::cout << "l.W.G" << l.W.G << std::endl;
    //std::cout << "l.b.G" << l.b.G << std::endl; 

    return 0;
}
