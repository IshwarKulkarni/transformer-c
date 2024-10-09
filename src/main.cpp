#include "../headers/functors.cuh"
#include "../headers/nodes.hpp"
#include "../headers/types"
#include <iostream>
#include "../headers/matrix.cuh"

using FloatT = float32;

int main(int argc, char const* argv[])
{
    // clang-format off
    FloatT xValues[] = {1.0f, 1.0f, 1.0f, 1.0f};
    FloatT wv[] = {0.254053, -0.247817, -0.427396, -0.014449, 
                    0.147829, 0.929250,  -0.807114,  -0.068262, 
                    0.218884, -0.453263, 0.800626,  -0.321338};

    FloatT bv[] = {-0.118935, -0.542531, 0.981391, -1.293095};
    // clang-format off

    Matrix<FloatT> x(3 , 1, xValues); // use only 3

    Linear<FloatT, Sigmoid<FloatT>> l(3, 4, wv, bv);
    std::cout << "x: " << x << std::endl;
    std::cout << "l.W : " << l.W.Weights << std::endl;
    std::cout << "l.b : " << l.b.Weights << std::endl;

    const auto& linear_out = l.forward(x);

    std::cout << "linear_out: " << linear_out << std::endl;

    MSE<FloatT> err(4, 1, false); 

    Matrix<FloatT> target(4, 1, xValues);

    const auto& out = err.forward(linear_out, target);

    std::cout << " out: " << out << std::endl;

    err.backward(out, target);

    //std::cout << "l.W.G" << l.W.G << std::endl;
    //std::cout << "l.b.G" << l.b.G << std::endl; 

    return 0;
}
