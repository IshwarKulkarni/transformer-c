#include <cuda_device_runtime_api.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "../headers/functors.cuh"
#include "../headers/matrix.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/types"

using FloatT = float32;

int main(int argc, char const* argv[])
{
    // clang-format off
    FloatT l1_W_Weights[] = {
        0.741299033165, -0.599930763245, -2.907784461975,
        -2.069497585297, 1.328607797623,  0.366938501596,
        -0.426573902369, -0.605450630188, -2.320645809174,
        -0.401497691870, -0.767400264740, -0.639289081097};

    FloatT l1_b_Weights[] = {
        0.78911542892,
        3.62128937244,
        -2.08241403103,
        0.55932860821};

    FloatT l2_W_Weights [] = {
        0.819633007050, -1.355650424957,  0.068369761109, -0.321855306625,
        -0.643309593201,  0.495646774769,  1.064129829407, -1.819470167160,
        -1.453269362450,  0.257461667061, -2.208420753479, -0.569770336151};

    FloatT l2_b_Weights[] = {
        2.83635449409,
        -1.8558882177,
        -2.53402024508};
    // clang-format off

    uint32_t in_vec_size = 3;
    uint32_t out_vec_size = 3;
    uint32_t hidden_size = 4;

    Matrix<FloatT> x(in_vec_size , 1); 
    Matrix<FloatT> t(out_vec_size, 1);
    fillCPU(x, FloatT(1.0));
    fillCPU(t, FloatT(.7));

    Linear<FloatT, Sigmoid<FloatT>> l1(in_vec_size, hidden_size);
    Linear<FloatT, Sigmoid<FloatT>> l2(hidden_size, out_vec_size, &l1);
    Linear<FloatT, Sigmoid<FloatT>> l3(out_vec_size,out_vec_size, &l2);
    L2Error<FloatT> errF(out_vec_size, 1, &l3);

    fill(l1.W.Weights, l1_W_Weights);
    fill(l1.b.Weights, l1_b_Weights);
    fill(l2.W.Weights, l2_W_Weights);
    fill(l2.b.Weights, l2_b_Weights);
    
    fillCPU(l3.W.Weights, FloatT(0.5));
    fillCPU(l3.b.Weights, FloatT(0.3)); 


    for (int i = 0; i < 1000; i++)
    {
        const auto& y1 = l1.forward(x);
        const auto& err = errF.forward(y1, t);

        if(i % 100 == 0)
        {
            std::cout << "Error: " << err(0, 0) << std::endl;
            std::cout << "y " << y1 << std::endl;
        }

        const auto& gradE = errF.backward(err);
        
        break;
        l1.update_weights(0.1);        
    }
    cudaErrCheck(cudaDeviceSynchronize());

    FloatT in = 0.5;
    fillCPU(x, in);
    const auto& out = l1.forward(x);

    std::cout
        << " l1.W grads: " << l1.W.Grads
        << " l1.b grads: " << l1.b.Grads
        << std::endl << in << " out " << out << std::endl;

    return 0;
}

/*
W1.grad
 tensor([[7.8610801211e-06, 7.8610801211e-06, 7.8610801211e-06],
        [6.2413355408e-06, 6.2413355408e-06, 6.2413355408e-06],
        [5.0871754809e-08, 5.0871754809e-08, 5.0871754809e-08],
        [7.3763418186e-05, 7.3763418186e-05, 7.3763418186e-05]])
b1.grad
 tensor([[7.8610801211e-06],
        [6.2413355408e-06],
        [5.0871754809e-08],
        [7.3763418186e-05]])
**/