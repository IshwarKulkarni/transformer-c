#include "../headers/functors.cuh"
#include "../headers/nodes.hpp"
#include "../headers/types"
#include <iostream>

int main(int argc, char const* argv[])
{
    float32 xValues[] = {1.0f, 1.0f, 1.0f};
    Matrix<float32> x(3, 1, xValues);

    Linear<float32, Sigmoid<float32>> l(3, 4);
    std::cout << "l.W.W : " << l.W.W << std::endl;
    std::cout << "l.b.W : " << l.b.W << std::endl;

    MSE<float32> err(4, 1, false);

    const auto& out = err.forward(l.forward(x), x);

    std::cout << out << std::endl;
    return 0;
}
