#include <cstdlib>
#include <iostream>
#include "../headers/matrix.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/types"

using FloatT = float64;

FloatT Ws [] = {
0.196004778147, -0.364338934422, -0.289773404598,  0.516708076000,  0.321904510260,
1.223316550255, -0.764124929905, -0.328112542629, -0.325616359711,  0.064051106572,
0.728902995586,  0.694612324238,  1.021002888680, -0.199148222804, -1.165511012077};

FloatT bs[] = {
0.351106971502,
0.010815961286,
0.052429087460};

FloatT bG[] = {
0.000356434665,
-0.002654204703,
0.001664445777,
};

int main()
{
    uint32 _in = 5;
    uint32 _out = 3;

    Matrixf x(_in, 1);
    LinearSigmoidF L0(_in, _out);
    SoftmaxF S(_out, 1, &L0);
    L2ErrorF mse(_out, 1, &S);

    L0.W.fill_values(Ws);
    L0.b.fill_values(bs);

    fillCPU(x, (FloatT)(1));
    Matrixf t(_out, 1);
    fillCPU(t, FloatT(0.7));

    mse.forward(L0.forward(x), t);
    mse.backward(t);

    std::cout << L0.b;

    if(!sameCPU(L0.b.Grads, bG, 1e-6))
    {
        std::cout << "Test failed" << std::endl;
        return 1;
    }
    std::cout << GREEN << "Test passed" << std::endl;

    return 0;
}
