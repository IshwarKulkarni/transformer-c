#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "../headers/loss_nodes.hpp"
#include "../headers/matrix.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/types"

using FloatT = float64;

// clang-format off
static const
FloatT Ws[] = {0.196004778147, -0.364338934422, -0.289773404598, 
               0.516708076000,  0.321904510260,  1.223316550255, 
              -0.764124929905, -0.328112542629, -0.325616359711,
               0.064051106572,  0.728902995586,  0.694612324238,
               1.021002888680, -0.199148222804, -1.165511012077};
static const
FloatT bs[] = {0.351106971502,  0.010815961286,  0.052429087460, .25, .33};
// clang-format on

int sizes_mismatch(std::string test_name, Matrix<FloatT>& matrix, const FloatT* expected,
                   FloatT eps = 1e-6)
{
    cudaErrCheck(cudaDeviceSynchronize());
    if (!sameCPU(matrix, expected, eps))
    {
        LOG(RED, BOLD, "Test ", test_name, " failed to match");
        LOG(RED, matrix, RESET, " and with expected");
        LOG(GREEN, shaped_like(matrix, expected));
        return 1;
    }
    LOG(GREEN, "Test ", test_name, " passed");
    return 0;
}

int test_Linear()
{
    LinearSigmoidF L0(5, 3, 2);
    LinearSigmoidF L1(3, 5, 2, true, &L0);
    CrossEntropyLoss<FloatT> l1e(5, 2, &L1);

    // std::cout << L0.graph_rep();

    L0.W.fill_values(Ws);
    L0.b.fill_values(bs);

    L1.W.fill_values(Ws);
    L1.b.fill_values(bs);

    Matrixf x(5, 2);
    fillCPU(x, (FloatT)(1));

    Matrixf t(5, 2);
    fillCPU(t, FloatT(0.7));

    const auto* y = L0.forward(&x);
    const auto* err = l1e.forward(y, &t);
    l1e.backward();

    FloatT b0G[] = {-0.0520891696, 0.0928521454, 0.1384243667};
    FloatT W1G[] = {-0.4506471455, -0.3137889206, -0.5047801733, -0.1816163212, -0.1264607757,
                    -0.2034325898, -0.6649558544, -0.4630136192, -0.7448322177, -0.2253836691,
                    -0.1569363028, -0.2524574101, -0.4621480405, -0.3217970729, -0.5176625848};

    return sizes_mismatch("Linear-L0.b", L0.b.Grads, b0G) +
           sizes_mismatch("Linear-L1.W", L1.W.Grads, W1G);
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

    const auto* y = L0.forward(&x);
    const auto* err = l1e.forward(y, &t);
    l1e.backward();

    FloatT bG[] = {-0.0731064528, -0.0830356926, 0.0614434481};
    return sizes_mismatch("L1", L0.b.Grads, bG);
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

    const auto* y = L0.forward(&x);
    const auto* err = mse.forward(y, &t);

    mse.backward();
    FloatT bG[] = {0.0694743693, 0.0448588878, 0.0683571771};
    return sizes_mismatch("L2", L0.b.Grads, bG);
}

int test_SMCE(uint32 _xW = 1)
{
    if (_xW > 3) throw std::runtime_error("Test for CE failed for _xW >= 3\n\n");

    uint32 _xH = 5;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    SoftmaxF S(_tH, _tW, &L0);
    CrossEntropyLoss<FloatT> ce(_tH, _tW, &S);

    L0.W.fill_values(Ws);
    L0.b.fill_values(bs);

    Matrixf x(_xH, _xW);
    fillCPU(x, (FloatT)(1));
    Matrixf t(_tH, _tW);

    FloatT probs[3][9] = {{1.0, 1.0, 1.0},
                          {0.1, 0.9, 0.3, 0.7, 0.6, 0.4},
                          {0.1, 0.6, 0.3, 0.7, 0.1, 0.2, 0.2, 0.3, 0.5}};

    fillCPU(t, probs[_xW - 1]);

    const auto* y = L0.forward(&x);
    const auto* ce_out = ce.forward(y, &t);
    ce.backward();

    FloatT bG_xw[3] = {0.0076335, -0.0391180, 0.0225302};
    return sizes_mismatch("SMCE" + std::to_string(_xW), L0.b.Grads, bG_xw);
}

int test_sm()
{
    uint32 _xH = 5;
    uint32 _xW = 2;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    SoftmaxF S(_tH, _tW, &L0);
    L2Loss<FloatT> error(_tH, _xW, &S);

    L0.W.fill_values(Ws);
    L0.b.fill_values(bs);

    Matrixf x(_xH, _xW);
    FloatT xVals[] = {3, .3, 3, .1, 2, .1, 2, .1, 2, .3};
    fillCPU(x, xVals);

    Matrixf t(_tH, _tW);
    FloatT tVals[] = {5, 3, 10, 5, 6, 4};
    fillCPU(t, tVals);

    const auto* y = L0.forward(&x);
    const auto* e = error.forward(y, &t);
    error.backward();

    FloatT L0WG[] = {0.1198487431,  0.1143481433,  0.0771488622,  0.0771488622,  0.0826494619,
                     -0.2293598950, -0.2238021642, -0.1501277387, -0.1501277387, -0.1556854695,
                     0.0082313018,  0.0084235799,  0.0055836732,  0.0055836732,  0.0053913947};
    FloatT L0bG[] = {0.0647022724, -0.1014631093, 0.0018785134};

    if (sizes_mismatch("SM_Wg", L0.W.Grads, L0WG) or sizes_mismatch("SM_bg", L0.b.Grads, L0bG))
    {
        return 1;
    }
    return 0;
}

int test_attention()
{
    uint32 emb_size = 3;
    uint32 seq_len = 2;
    Attention<FloatT> att(emb_size, seq_len);

    att.Wq.W.fill_value(0.1);
    att.Wk.W.fill_value(0.1);
    att.Wv.W.fill_value(0.1);

    Matrixf x(emb_size, seq_len);
    FloatT xVals[] = {1, 2, 3, 4, 5, 6};
    fillCPU(x, xVals);

    const auto* y = att.forward(&x);
    return 0;
}

int main(int argc, char const* argv[])
{
    test_Linear();
    test_sm();
    test_L2();
    test_L1();
    test_SMCE(1);
    test_SMCE(2);
    test_SMCE(3);
    // test_attention();
    return 0;
}