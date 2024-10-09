#include <cstdlib>
#include <iostream>
#include "../headers/matrix.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/types"

using FloatT = float64;

int main(int argc, char const* argv[])
{
    uint32 _in = 5;
    uint32 _out = 3;

    Matrixf x(_in, 1);
    LinearIdentityF L(_in, _out);
    SoftmaxF S(_out, 1, &L);
    L2ErrorF mse(_out, 1, &S);

    fillCPU(x, (FloatT)(1));
    fillCPU(L.W.Weights, FloatT(0.1));
    fillCPU(L.b.Weights, FloatT(0.1));

    const auto& y = L.forward(x);

    Matrixf t(_out, 1);
    const auto& e = mse.forward(y, t);
    fillCPU(t, FloatT(0.7));

    cudaErrCheck(cudaDeviceSynchronize());
    std::cout << "lin out: " << L.get_output() << std::endl
              << "sm exps: " << S.exps << std::endl
              << "sm sum_0d: " << S.temp0d << std::endl
              << "softmax out: " << S.get_output() << std::endl
              << "y: " << y << std::endl
              << "mse out: " << mse.get_output() << std::endl;

    mse.backward(e);
    cudaErrCheck(cudaDeviceSynchronize());
    std::cout << L.W.Weights << std::endl << L.W.Grads << std::endl;
}
