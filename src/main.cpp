#include <cuda_device_runtime_api.h>
#include <cstdlib>
#include <stdexcept>
#include "../headers/loss_nodes.hpp"
#include "../headers/matrix.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/types"
#include "../headers/word2vec.hpp"

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

int values_mismatch(std::string test_name, Matrix<FloatT>& matrix, const FloatT* expected,
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

    fill(L0.W, Ws);
    fill(L0.b, bs);
    fill(L1.W, Ws);
    fill(L1.b, bs);

    Matrixf x(5, 2);
    fillCPU(x, (FloatT)(1));

    Matrixf t(5, 2);
    fillCPU(t, FloatT(0.7));

    const auto* y = L0.forward(&x);
    l1e.forward(y, &t);
    l1e.backward();

    FloatT b0G[] = {-0.0520891696, 0.0928521454, 0.1384243667};
    FloatT W1G[] = {-0.4506471455, -0.3137889206, -0.5047801733, -0.1816163212, -0.1264607757,
                    -0.2034325898, -0.6649558544, -0.4630136192, -0.7448322177, -0.2253836691,
                    -0.1569363028, -0.2524574101, -0.4621480405, -0.3217970729, -0.5176625848};

    return values_mismatch("Linear-L0.b", L0.b.grads, b0G) +
           values_mismatch("Linear-L1.W", L1.W.grads, W1G);
}

int test_L1()
{
    uint32 _xH = 5;
    uint32 _xW = 2;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    L1Loss<FloatT> l1e(_tH, _xW, &L0);

    fill(L0.W, Ws);
    fill(L0.b, bs);

    Matrixf x(_xH, _xW);
    fillCPU(x, (FloatT)(1));

    Matrixf t(_tH, _tW);
    fillCPU(t, FloatT(0.7));

    const auto* y = L0.forward(&x);
    l1e.forward(y, &t);
    l1e.backward();

    FloatT bG[] = {-0.0731064528, -0.0830356926, 0.0614434481};
    return values_mismatch("L1", L0.b.grads, bG);
}

int test_L2()
{
    uint32 _xH = 5;
    uint32 _xW = 2;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    L2Loss<FloatT> mse(_tH, _xW, &L0);

    fill(L0.W, Ws);
    fill(L0.b, bs);

    Matrixf x(_xH, _xW);
    fillCPU(x, (FloatT)(1));

    Matrixf t(_tH, _tW);
    fillCPU(t, FloatT(0.2));

    const auto* y = L0.forward(&x);
    mse.forward(y, &t);

    mse.backward();
    FloatT bG[] = {0.0694743693, 0.0448588878, 0.0683571771};
    return values_mismatch("L2", L0.b.grads, bG);
}

int test_SMCE(uint32 _xW = 1)
{
    if (_xW > 2) throw std::runtime_error("Test for CE failed for _xW >= 2\n\n");

    uint32 _xH = 5;
    uint32 _tW = _xW;
    uint32 _tH = 3;

    LinearSigmoidF L0(_xH, _tH, _xW);
    SoftmaxDim0F S(_tH, _tW, &L0);
    CrossEntropyLoss<FloatT> ce(_tH, _tW, &S);

    fill(L0.W, Ws);
    fill(L0.b, bs);

    Matrixf x(_xH, _xW);
    fillCPU(x, (FloatT)(1));
    Matrixf t(_tH, _tW);

    FloatT probs[3][9] = {{1.0, 1.0, 1.0},
                          {0.1, 0.9, 0.3, 0.7, 0.6, 0.4},
                          {0.1, 0.6, 0.3, 0.7, 0.1, 0.2, 0.2, 0.3, 0.5}};

    fillCPU(t, probs[_xW - 1]);

    const auto* y = L0.forward(&x);
    ce.forward(y, &t);
    ce.backward();

    FloatT bG_xw[3] = {0.0076335, -0.0391180, 0.0225302};
    return values_mismatch("SMCE" + std::to_string(_xW), L0.b.grads, bG_xw);
}

int test_sm()
{
    uint32 _xH = 5;
    uint32 _xW = 2;
    uint32 _tH = 3;
    uint32 _tW = _xW;

    LinearSigmoidF L0(_xH, _tH, _xW);
    SoftmaxDim0F S(_tH, _tW, &L0);
    L2Loss<FloatT> error(_tH, _xW, &S);

    fill(L0.W, Ws);
    fill(L0.b, bs);

    Matrixf x(_xH, _xW);
    FloatT xVals[] = {3, .3, 3, .1, 2, .1, 2, .1, 2, .3};
    fillCPU(x, xVals);

    Matrixf t(_tH, _tW);
    FloatT tVals[] = {5, 3, 10, 5, 6, 4};
    fillCPU(t, tVals);

    const auto* y = L0.forward(&x);
    error.forward(y, &t);
    error.backward();

    FloatT L0WG[] = {0.1198487431,  0.1143481433,  0.0771488622,  0.0771488622,  0.0826494619,
                     -0.2293598950, -0.2238021642, -0.1501277387, -0.1501277387, -0.1556854695,
                     0.0082313018,  0.0084235799,  0.0055836732,  0.0055836732,  0.0053913947};
    FloatT L0bG[] = {0.0647022724, -0.1014631093, 0.0018785134};

    if (values_mismatch("SM_Wg", L0.W.grads, L0WG) or values_mismatch("SM_bg", L0.b.grads, L0bG))
    {
        return 1;
    }
    return 0;
}

int test_word2vec()
{
    const char* filename = "/home/ishwark/word2vecdata//wiki.multi.en.vec";
    Word2Vec word2vec(filename);
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, word2vec.size() - 1);

    std::vector<std::string> sample_words = {
        "king",   "queen",  "knight",   "prince", "princess", "knight",   "castle",  "kingdom",
        "boy",    "girl",   "mom",      "dad",    "brother",  "uncle",    "aunt",    "grandma",
        "laptop", "mouse",  "keyboard", "screen", "monitor",  "cpu",      "gpu",     "ram",
        "table",  "chair",  "sofa",     "bed",    "lamp",     "fan",      "blanket", "tv",
        "bus",    "car",    "bike",     "train",  "plane",    "ship",     "boat",    "truck",
        "oak",    "maple",  "pine",     "birch",  "beech",    "mahogany", "teak",    "cedar",
        "dog",    "cat",    "fish",     "bird",   "rabbit",   "hamster",  "rat",     "mouse",
        "pen",    "pencil", "eraser",   "book",   "notebook", "diary",    "journal", "calendar"};

    std::string options[] = {"FAST", "ACCURATE", "EXACT"};

    std::vector<std::tuple<std::string, std::string, float32, float32>> sig033fast;

    for (FloatT sigma : {0.01, 0.033, 0.1})  // 10deg, 30deg, 60deg
    {
        LOG("");
        for (SearchOption option : {FAST, ACCURATE})
        {
            uint32 missed = 0;
            FloatT meanSim = 0;
            Timer timer("Word2Vec");
            uint32 tries = option == EXACT ? 1 : 4;
            word2vec.nearest_count = 0;
            for (uint32 i = 0; i < tries; ++i)
            {
                for (auto word : sample_words)
                {
                    auto node = word2vec[word];

                    Vec300 vec = node->vec;
                    auto sim = add_noise_normal(vec, 0, sigma);
                    meanSim += sim;
                    auto nearest = word2vec(vec, option);

                    if (*nearest == *node) continue;

                    if (sigma > 0.01 and sigma < 0.1 and option == FAST)
                        sig033fast.push_back({word, nearest->word, sim, nearest->cos_sim(node)});
                    // LOG(RED, word, " -> ", nearest->word, BLUE, " error ", sim,
                    //     "  |nearest-node|=", std::sqrt(nearest->dist2(node)));
                    //  print_path(node);
                    //  print_path(nearest);
                    missed++;
                }
            }
            uint32 total = tries * sample_words.size();
            FloatT success_rate = 100.0 * (total - missed) / total;
            FloatT nearest_count = FloatT(word2vec.nearest_count) / total;
            // clang-format off
            LOG(" Option     : ", std::setw(10), options[option], "\tsigma: ", std::setw(5), std::setprecision(4), sigma,
                " Success    : ", GREEN, std::setw(6), std::setprecision(4), success_rate, "%", RESET, 
                " Avg. calls : ", GREEN, std::setw(6), std::setprecision(6), nearest_count, RESET, 
                " Avg. time  : ", GREEN, std::setw(6), std::setprecision(4), timer.stop() * 1e3 / total, "ms.", RESET,
                " Avg. sim   : ", BLUE,  std::setw(6), std::setprecision(4), meanSim / total, RESET);
            // clang-format on
        }
    }
    // for (auto [word, nearest, sim, nearestSim] : sig033fast)
    //{
    //     LOG(RED, word, " -> ", nearest, BLUE, " error ", sim, "  |nearest-node| = ", nearestSim);
    // }
    return 0;
}

int test_attention()
{
    uint32 emb_size = 7;
    uint32 seq_len = 5;

    Node<FloatT>* prevs[] = {nullptr, nullptr, nullptr};

    Attention<FloatT> att(emb_size, seq_len, prevs);
    L2ErrorF error(seq_len, seq_len, &att);

    Matrixf qkv[] = {Matrixf(emb_size, seq_len), Matrixf(emb_size, seq_len),
                     Matrixf(emb_size, seq_len)};

    for (uint32 i = 0; i < 3; i++) fillCPU(qkv[i], FloatT(i + 1) / 10);

    // clang-format off
    FloatT QW[] = {
    -0.81733400, -0.55556852, -0.82668954, -1.29695439, -0.19737878, -0.96433353, -0.51329893,
    2.62778473, -0.74649578,  1.00509381, -0.25683922,  0.47649285, -0.66521013, -0.36266556,
    -1.45035112, -0.24958687,  0.82977319,  1.12094724,  0.99991328, -0.43443865, -2.18059897,
    -1.10937488, -2.04103804,  0.03336523,  1.62941325, -2.11841989,  0.78275818,  0.56316656,
    -1.11511636,  0.14900762, -1.09230065, -1.45512831,  1.33581829,  0.30752665,  0.62777656};

    FloatT KW[] = {
    2.74700475,  0.36901963,  1.33733141, -0.91797650, -0.96151292, -0.88965815,  1.06175268,
    -0.32060274, -0.76655698,  0.43009135,  0.14965299, -0.24599971, -1.46364880, -0.77558517,
    -0.25944236, -1.97904468, -0.57491088,  0.58530539, -1.36581850, -0.21945974, -1.72055340,
    -0.10526190, -0.47669813,  0.02417080,  1.02715981, -1.69959307,  0.24316004,  0.90815562,
    -2.47583365,  0.18871087, -0.70371968,  0.64837265,  2.62271261,  0.13389370, -0.86557102};

    FloatT VW[] = {
    -0.27535307, -0.37029237,  0.33241835,  0.52227819,  0.60274231, -0.54026198, -0.89765978,
    -0.09792950, -2.26444221, -1.32238948, -0.58601892,  1.61030805, -1.41811168,  1.48550093,
    0.07093975,  1.05998838,  0.78130090, -1.41996562,  2.14789987, -2.23898602,  0.42987868,
    -0.14024276, -0.36814091, -0.04156580,  2.39554644, -0.15455456, -0.07250153, -1.00988662,
    -0.16480856,  0.19279088, -0.62133241,  2.79566097,  0.50273341, -0.05686535,  1.39163923};

    FloatT Q_grad[] = {
    -0.00228311, -0.00228311, -0.00228311, -0.00228311, -0.00228311, -0.00228311, -0.00228311,
    -0.00056598, -0.00056598, -0.00056598, -0.00056598, -0.00056598, -0.00056598, -0.00056598,
    -0.00245539, -0.00245539, -0.00245539, -0.00245539, -0.00245539, -0.00245539, -0.00245539,
    -0.00264587, -0.00264587, -0.00264587, -0.00264587, -0.00264587, -0.00264587, -0.00264587,
    -0.00241836, -0.00241836, -0.00241836, -0.00241836, -0.00241836, -0.00241836, -0.00241836};

    FloatT K_grad[] = {
    -0.00224513, -0.00224513, -0.00224513, -0.00224513, -0.00224513, -0.00224513, -0.00224513,
    -0.02348177, -0.02348177, -0.02348177, -0.02348177, -0.02348177, -0.02348177, -0.02348177,
    0.00590191,  0.00590191,  0.00590191,  0.00590191,  0.00590191,  0.00590191,  0.00590191,
    0.00125742,  0.00125742,  0.00125742,  0.00125742,  0.00125742,  0.00125742,  0.00125742,
    0.01856757,  0.01856757,  0.01856757,  0.01856757,  0.01856757,  0.01856757,  0.01856757};

    FloatT V_grad[] = {
    -0.08461417, -0.08461417, -0.08461417, -0.08461417, -0.08461417, -0.08461417, -0.08461417,
    -0.11256053, -0.11256053, -0.11256053, -0.11256053, -0.11256053, -0.11256053, -0.11256053,
    -0.13796328, -0.13796328, -0.13796328, -0.13796328, -0.13796328, -0.13796328, -0.13796328,
    -0.09446066, -0.09446066, -0.09446066, -0.09446066, -0.09446066, -0.09446066, -0.09446066,
    -0.09626430, -0.09626430, -0.09626430, -0.09626430, -0.09626430, -0.09626430, -0.09626430};
    // clang-format on

    fillCPU(att.Q.W, QW);
    fillCPU(att.K.W, KW);
    fillCPU(att.V.W, VW);

    SoftmaxDim0F softmax(seq_len, seq_len, nullptr, "Softmax");
    Matrixf t = ones<FloatT>(seq_len, seq_len);

    const auto* y = att.forward(qkv);
    error.forward(y, &t);
    error.backward();

    return values_mismatch("Q_grad", att.Q.W.grads, Q_grad) +
           values_mismatch("K_grad", att.K.W.grads, K_grad) +
           values_mismatch("V_grad", att.V.W.grads, V_grad);
}

int main()
{
    // test_Linear();
    // test_sm();
    // test_L2();
    // test_L1();
    // test_SMCE(1);
    // test_SMCE(2);
    test_attention();
    // test_word2vec();
    return 0;
}