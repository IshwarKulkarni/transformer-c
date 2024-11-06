#include "../headers/loss_nodes.hpp"
#include "../headers/matrix_ops.hpp"
#include "../headers/nodes.hpp"
#include "../headers/word2vec.hpp"
#include "fstream"

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
    return 0;
}

int test_fc()
{
    uint32 xh = 3;
    uint32 xw = 4;
    uint32 th = 2;
    uint32 inner = 5;
    Input<> x(xh, xw, "x");

    FullyConnected<FloatT, Sigmoid<FloatT>> L0(xh, inner, xw, {&x}, true, "Linear-L0");
    FullyConnected<FloatT, Sigmoid<FloatT>> L1(inner, th, xw, {&L0}, true, "Linear-L1");
    SoftmaxDim0<FloatT> S({&L1}, "Softmax-Dim0");
    Input<> t(S.shape(), "target");
    CrossEntropyLoss<FloatT> loss({&t, &S}, "L2Error");

    fillCPU(x, 1);
    fillCPU(t, 5);

    Matrix<FloatT> W0_grad(L0.W.shape());
    Matrix<FloatT> b0_grad(L0.b.shape());
    Matrix<FloatT> W1_grad(L1.W.shape());
    Matrix<FloatT> b1_grad(L1.b.shape());

    std::ifstream golden("static_data/fc.txt");
    golden >> L0.W >> L0.b >> L1.W >> L1.b >> W0_grad >> b0_grad >> W1_grad >> b1_grad;

    loss.forward();
    loss.backward();

    uint32 err = values_mismatch("Linear-L0.W", L0.W.grads, W0_grad.begin()) +
                 values_mismatch("Linear-L1.W", L1.W.grads, W1_grad.begin()) +
                 values_mismatch("Linear-L1.b", L1.b.grads, b1_grad.begin());

    if (err == 0) LOG(GREEN, "Test FC passed");
    return err;
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
    Input<> q_(emb_size, seq_len, "q_input");
    Input<> k_(emb_size, seq_len, "k_input");
    Input<> v_(emb_size, seq_len, "v_input");
    fillCPU(q_, .1);
    fillCPU(k_, .2);
    fillCPU(v_, .3);

    Attention<> attention(emb_size, seq_len, {&q_, &k_, &v_}, "Attention");

    Matrix<FloatT> Q_grad(attention.Q.W.shape());
    Matrix<FloatT> K_grad(attention.K.W.shape());
    Matrix<FloatT> V_grad(attention.V.W.shape());

    std::ifstream golden("static_data/attention.txt");
    golden >> attention.Q.W >> attention.K.W >> attention.V.W;
    golden >> Q_grad >> K_grad >> V_grad;

    Input<> target(seq_len, seq_len, "target");
    fillCPU(target, (1.));

    L2Loss<> l2({&target, &attention});
    std::ofstream out("graph.dot");
    graph_to_dot(&l2, out);

    l2.forward();
    l2.backward();

    uint32 err = values_mismatch("Q_grad", attention.Q.W.grads, Q_grad.begin()) +
                 values_mismatch("K_grad", attention.K.W.grads, K_grad.begin()) +
                 values_mismatch("V_grad", attention.V.W.grads, V_grad.begin());
    if (err == 0) LOG(GREEN, "Test attention passed");
    return err;
}

int main()
{
    test_word2vec();
    test_fc();
    test_attention();
    return 0;
}
