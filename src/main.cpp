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
        // LOG(RED, matrix, RESET, " and with expected");
        // LOG(GREEN, expected);
        return 1;
    }
    else
    {
        LOG(YELLOW, test_name, " passed matching ", matrix.name, matrix.shape_str);
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
    FullyConnected<FloatT, TanH<FloatT>> L1(inner, th, xw, {&L0}, true, "Linear-L1");
    SoftmaxDim0<FloatT> S({&L1}, "Softmax-Dim0");
    Input<> t(S.shape(), "target");
    CrossEntropyLoss<FloatT> loss({&t, &S}, "L2Error");

    fillCPU(x, 1);
    fillCPU(t, 5);

    Matrix<FloatT> W0_grad = Matrix<FloatT>(L0.W.grads.shape());
    Matrix<FloatT> b0_grad = Matrix<FloatT>(L0.b.grads.shape());
    Matrix<FloatT> W1_grad = Matrix<FloatT>(L1.W.grads.shape());
    Matrix<FloatT> b1_grad = Matrix<FloatT>(L1.b.grads.shape());

    std::ifstream golden("static_data/fc.txt");
    golden >> L0.W >> L0.b >> L1.W >> L1.b >> W0_grad >> b0_grad >> W1_grad >> b1_grad;

    loss.compute();
    loss.backward();

    uint32 err = values_mismatch("Linear-L0.W", L0.W.grads, W0_grad) +
                 values_mismatch("Linear-L1.W", L1.W.grads, W1_grad) +
                 values_mismatch("Linear-L1.b", L1.b.grads, b1_grad);

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
    std::ifstream golden("static_data/attention.txt");
    uint32 Ei;  //  input embedding size
    uint32 Eq;  //  query embedding size
    golden >> Eq >> Ei;
    golden.seekg(0, std::ios::beg);
    uint32 Ev = Ei;  //  value, i.e. output embedding size
    uint32 S = 10;   //  sequence length

    Input<> q(S, Ei, "Query"), k(S, Ei, "Key"), v(S, Ei, "Value");

    Attention<FloatT> A(Eq, Ev, {&q, &k, &v}, "Attention");
    Input<> target(A.shape(), "target");
    fillCPU(target, 1);
    L2Loss<FloatT> loss({&A, &target}, "L2Error");

    golden >> A.Q.W >> A.K.W >> A.V.W;
    golden >> q >> k >> v;

    loss.compute();
    loss.backward();

    uint32 err = values_mismatch("Attn. qkt", A.qkT, read_csv<FloatT>(golden)) +
                 values_mismatch("Attn. smax", A.attention_weights, read_csv<FloatT>(golden)) +
                 values_mismatch("Attn. out", A.attention, read_csv<FloatT>(golden)) +
                 values_mismatch("Q.W.grads", A.Q.W.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("K.W.grads", A.K.W.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("V.W.grads", A.V.W.grads, read_csv<FloatT>(golden));

    if (err == 0) LOG(GREEN, "Test Attention passed");
    return err;
}

int test_ProductT()
{
    Input<> A(3, 4, "A");
    Input<> B(5, 4, "B");
    Input<> target(3, 5, "target");
    ProductT<FloatT, DividebBy<FloatT>> P({&A, &B}, DividebBy<FloatT>(5), "ProductT");

    L2Loss<FloatT> loss({&target, &P}, "L2Error");

    fillCPU(A, 1);
    fillCPU(B, 2);
    fillCPU(target, 3);
    loss.compute();
    loss.backward();
    return 0;
}

int test_linear()
{
    Input<> x(3, 4, "A");
    Linear<FloatT> L0(5, {&x}, "Linear0");
    Linear<FloatT> L1(6, {&L0}, "Linear1");

    std::ifstream golden("static_data/linear.txt");
    golden >> L0.W >> L1.W >> x;

    Input<> target(L1.shape(), "target");
    fillCPU(target, 1);
    L2Loss<FloatT> loss({&L1, &target}, "L2Error");

    loss.compute();
    loss.backward();

    uint32 err = values_mismatch("L0.W.grads", L0.W.grads, read_csv<FloatT>(golden)) +
                 values_mismatch("L1.W.grads", L1.W.grads, read_csv<FloatT>(golden));
    if (err == 0) LOG(GREEN, "Test Linear passed");
    return err;
}

int time_attention()
{
    uint32 Ei = 640;  //  input embedding size
    uint32 Eq = 128;  //  query embedding size
    uint32 Ev = Ei;   //  value, i.e. output embedding size for each head
    uint32 Sl = 20;   //  sequence length
    Input<> q(Sl, Ei, "Query"), k(Sl, Ei, "Key"), v(Sl, Ei, "Value");
    Attention<FloatT> A(Eq, Ev, {&q, &k, &v}, "Attention");
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
        float64(MatrixIds::get_alloced_bytes()) / (sizeof(FloatT) * (1 << 20)), "MB");
    return 0;
}

int test_multihead()
{
    uint32 Ei = 48;  //  input embedding size
    uint32 Eq = 16;  //  query embedding size
    uint32 Ev = 20;  //  value, i.e. output embedding size for each head
    uint32 Sl = 20;  //  sequence length
    uint32 Eo = Ei;  //  output embedding size, same as input

    Input<> q(Sl, Ei, "Query"), k(Sl, Ei, "Key"), v(Sl, Ei, "Value");
    MultiHeadAttention<FloatT> M(3, Eq, Ev, Eo, {&q, &k, &v}, "MultiHeadAttention");
    Input<> target(M.shape(), "target");
    fillCPU(target, 1);
    L2Loss<FloatT> loss({&M, &target}, "L2Error");

    loss.compute();
    loss.backward();

    return 0;
}

int test_dropout(uint32 h, uint32 w, FloatT p)
{
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
    if (sumB != sum or std::abs(FloatT(sum)/A.numels() + p - 1) > 0.02)
    {
        LOG(RED, "Dropout failed: ", sum, " vs ", sumB, " or ", sum/A.numels(), " vs ", p);
        return -1;
    }
    LOG(GREEN, "Dropout passed");
    return 0;
}

int main()
{
    //test_fc();
    //test_linear();
    //test_ProductT();
    //test_attention();
    //time_attention();
    //test_multihead();
    test_dropout(100, 100, 0.25);
    return 0;
}
