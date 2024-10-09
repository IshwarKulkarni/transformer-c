#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "../headers/matrix.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/types"

using FloatT = float64;

// clang-format off
FloatT Ws[] = {0.196004778147, -0.364338934422, -0.289773404598, 
               0.516708076000,  0.321904510260,  1.223316550255, 
              -0.764124929905, -0.328112542629, -0.325616359711,
               0.064051106572,  0.728902995586,  0.694612324238,
               1.021002888680, -0.199148222804, -1.165511012077};
FloatT bs[] = {0.351106971502,  0.010815961286,  0.052429087460};
// clang-format on

int check_gradients(std::string test_name, Matrix<FloatT>& grads, const FloatT* bG, FloatT eps = 1e-6)
{
    cudaErrCheck(cudaDeviceSynchronize());
    if (!sameCPU(grads, bG, eps))
    {
        LOG(RED, BOLD, "Test " , test_name , " failed");
        std::cout << grads << std::endl;
        return 1;
    }
    LOG(GREEN, "Test ", test_name, " passed");
    return 0;
}

int test_L1()
{
    uint32 _xH = 5;
    uint32 _xW = 2;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    L1Loss<FloatT> l1e(_tH, _xW, &L0);

    L0.W.fill_values(Ws);
    L0.b.fill_values(bs);

    Matrixf x(_xH, _xW);
    fillCPU(x, (FloatT)(1));

    Matrixf t(_tH, _tW);
    fillCPU(t, FloatT(0.7));

    const auto& y = L0.forward(x);
    const auto& err = l1e.forward(y, t);
    l1e.backward();

    FloatT bG[] = {-0.0731064528, -0.0830356926, 0.0614434481};
    return check_gradients("L1",  L0.b.Grads, bG);
}

int test_L2()
{
    uint32 _xH = 5;
    uint32 _xW = 2;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    L2Loss<FloatT> mse(_tH, _xW, &L0);

    L0.W.fill_values(Ws);
    L0.b.fill_values(bs);

    Matrixf x(_xH, _xW);
    fillCPU(x, (FloatT)(1));

    Matrixf t(_tH, _tW);
    fillCPU(t, FloatT(0.2));

    const auto& y = L0.forward(x);
    const auto& err = mse.forward(y, t);

    mse.backward();
    FloatT bG[] = {0.0694743693, 0.0448588878, 0.0683571771};
    return check_gradients("L2", L0.b.Grads, bG);
}

int test_SMCE(uint32 _xW = 1)
{
    if (_xW > 3) throw std::runtime_error("Test for CE failed for _xW >= 3\n\n");

    uint32 _xH = 5;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    SoftmaxF S(_tH, _tW, true, &L0);
    CrossEntropyLoss<FloatT> ce(_tH, _tW, &S);

    L0.W.fill_values(Ws);
    L0.b.fill_values(bs);

    Matrixf x(_xH, _xW);
    fillCPU(x, (FloatT)(1));
    Matrixf t(_tH, _tW);

    FloatT probs[][9] = {{1.0, 1.0, 1.0},
                         {0.1, 0.9, 0.3, 0.7, 0.6, 0.4},
                         {0.1, 0.6, 0.3, 0.7, 0.1, 0.2, 0.2, 0.3, 0.5}};

    fillCPU(t, probs[_xW - 1]);

    const auto& y = L0.forward(x);
    const auto& ce_out = ce.forward(y, t);
    ce.backward();

    // BG remains same because probs add to 1 row wise
    FloatT bG_xw[3] = {0.0076335, -0.0391180, 0.0225302};
    return check_gradients("SMCE" + std::to_string(_xW), L0.b.Grads, bG_xw);
}

int main()
{
    test_L2();
    test_L1();
    test_SMCE(1);
    test_SMCE(2);
    test_SMCE(3);
    return 0;
}