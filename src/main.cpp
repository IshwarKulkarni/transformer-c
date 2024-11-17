#include "../headers/learning_nodes.hpp"
#include "../headers/loss_nodes.hpp"
#include "../headers/matrix_ops.hpp"
#include "../headers/word2vec.hpp"
#include "fstream"

int values_mismatch(std::string test_name, Matrix<FloatT>& matrix, const Matrix<FloatT>& expected,
                    FloatT eps = FloatT(1e-6))
{
    cudaErrCheck(cudaDeviceSynchronize());
    uint32 mismatches = sameCPU(matrix, expected, eps);
    if (mismatches)
    {
        LOG(RED, BOLD, test_name, " mismatch at ", mismatches, " locations, for  ", matrix.name,
            matrix.shape_str, " with eps: ", eps);
        LOG(RED, matrix, RESET, " and with expected");
        LOG(GREEN, expected);
        return 1;
    }
    return 0;
}

int test_word2vec(const char* filename)
{
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

int test_linearb()
{
    uint32 Ei = 3;
    uint32 Sl = 4;
    uint32 I1 = 5;
    uint32 I2 = 6;
    Input<> x(Sl, Ei, "x");

    Linear<FloatT, Sigmoid<FloatT>> L0(LinearInputT<FloatT>{I1, &x, true, "Linear-L0"});
    Linear<FloatT, TanH<FloatT>> L1(LinearInputT<FloatT>{I2, &L0, true, "Linear-L1"});

    std::ifstream golden("static_data/linearb.txt");
    golden >> x >> L0.W >> L0.b >> L1.W >> L1.b;

    Input<> t(L1.shape(), "target");
    fillCPU(t, 1);
    CrossEntropyLoss<> loss({&L1, &t}, "L2Error");

    loss.compute();
    loss.backward();
    cudaErrCheck(cudaDeviceSynchronize());

    uint32 err = values_mismatch("Output", L1, read_csv<FloatT>(golden)) +
                 values_mismatch("L1.W.grads", L1.W.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("L1.b.grads", L1.b.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("L0.W.grads", L0.W.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("L0.b.grads", L0.b.grads, read_csv<FloatT>(golden));

    if (err == 0) LOG(GREEN, "Test LinearBiasAct passed");
    return err;
}

int test_attention()
{
    std::ifstream golden("static_data/attention.txt");
    uint32 Ei;  //  input embedding size
    uint32 Eq;  //  query embedding size
    golden >> Eq >> Ei;
    golden.seekg(0, std::ios::beg);
    uint32 Ev = Ei;  //  value, i.e. output embedding size
    uint32 S = 10;   //  sequence length

    Input<> q(S, Ei, "Qi"), k(S, Ei, "Ki"), v(S, Ei, "Vi");

    Attention<> A({Eq, &q, false, "Attention_Q"}, {Eq, &k, false, "Attention_K"},
                  {Ev, &v, false, "Attention_V"}, "Attention");
    Input<> target(A.shape(), "target");
    fillCPU(target, 1);
    L2Loss<> loss({&A, &target}, "L2Error");

    golden >> A.Q.W >> A.K.W >> A.V.W >> q >> k >> v;

    loss.compute();
    loss.backward();

    std::ofstream dot("attention.dot");
    graph_to_dot(&loss, dot);

    uint32 err = values_mismatch("Attn. qkt", A.qkT, read_csv<FloatT>(golden)) +
                 values_mismatch("Attn. smx", A.attention_weights, read_csv<FloatT>(golden)) +
                 values_mismatch("Attn. out", A.attention, read_csv<FloatT>(golden)) +
                 values_mismatch("Q.W.grads", A.Q.W.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("K.W.grads", A.K.W.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("V.W.grads", A.V.W.grads, read_csv<FloatT>(golden));

    if (err == 0) LOG(GREEN, "Test Attention passed");
    return err;
}

int time_attention()
{
    uint32 Ei = 640;  //  input embedding size
    uint32 Eq = 128;  //  query embedding size
    uint32 Ev = Ei;   //  value, i.e. output embedding size for each head
    uint32 Sl = 20;   //  sequence length
    Input<> q(Sl, Ei, "Qi"), k(Sl, Ei, "Ki"), v(Sl, Ei, "Vi");

    // clang-format off
    Attention<> A({Eq, &q, false, "Attention_Q"}, 
                  {Eq, &k, false, "Attention_K"},
                  {Ev, &v, false, "Attention_V"}, 
                  "Attention");
    // clang-format on

    Input<> target(A.shape(), "target");
    L2Loss<FloatT> loss({&A, &target}, "L2Error");

    uint32 max_iters = 100;
    CudaEventTimer timer("Attention");
    for (uint32 i = 0; i < max_iters; ++i)
    {
        loss.compute();
        loss.backward();
    }
    cudaErrCheck(cudaDeviceSynchronize());
    LOG(GREEN, "Time per iteration: ", timer.stop() / max_iters, "ms");
    LOG(BLUE, "Allocated memory: ",
        float64(MatrixInitUitls::get_alloced_bytes()) / (sizeof(FloatT) * (1 << 20)), "MB");
    return 0;
}

int test_multihead()
{
    uint32 Ei = 6;   //  input embedding size
    uint32 Eq = 4;   //  query embedding size
    uint32 Ev = 7;   //  value, i.e. output embedding size for each head
    uint32 Sl = 5;   //  sequence length
    uint32 Eo = Ei;  //  output embedding size, same as input

    Input<> q(Sl, Ei, "Query"), k(Sl, Ei, "Key"), v(Sl, Ei, "Value");
    using Sig = Sigmoid<FloatT>;
    using Tan = TanH<FloatT>;
    using MHA = MultiHeadAttention<FloatT, Sig, Sig, Sig, Tan>;
    xavier_init<FloatT>(q);
    xavier_init<FloatT>(k);
    xavier_init<FloatT>(v);

    // clang-format off
    MHA M(3, {Eq, &q, true, "Q"},
             {Eq, &k, true, "K"},
             {Ev, &v, true, "V"},
             {Eo, nullptr, true, "O"},
             "MultiHeadAttention");
    // clang-format on

    Input<> target(M.shape(), "target");
    fillCPU(target, 1);
    L2Loss<FloatT> loss({&M, &target}, "L2Error");

    loss.compute();
    loss.backward();

    M.print_desc();

    std::ofstream dot("multihead.dot");
    graph_to_dot(&loss, dot);

    return 0;
}

int test_dropout(FloatT p)
{
    uint32 h = 100, w = 100;
    Matrix<FloatT> A(h, w);
    fillCPU(A, 1);
    Matrix<bool> mask(A.shape());
    dropout(A, mask, p);

    cudaErrCheck(cudaDeviceSynchronize());
    FloatT sum = std::accumulate(A.begin(), A.end(), 0);

    Matrix<FloatT> B(h, w);
    fillCPU(B, 1);
    dropout(B, mask, -1);
    cudaErrCheck(cudaDeviceSynchronize());
    uint32 sumB = std::accumulate(B.begin(), B.end(), 0);
    if (sumB != sum or std::abs(FloatT(sum) / A.numels() + p - 1) > 0.02)
    {
        LOG(RED, "Dropout failed: ", sum, " vs ", sumB, " or ", sum / A.numels(), " vs ", p);
        return -1;
    }
    LOG(GREEN, "Dropout passed");
    return 0;
}

int main()
{
    // test_word2vec("/home/ishwark/word2vecdata/wiki.multi.en.vec");
    test_attention();
    test_multihead();
    //  test_dropout(0.25);
    test_linearb();
    // time_attention();
    return 0;
}
