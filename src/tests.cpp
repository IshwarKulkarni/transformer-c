#include <cuda_device_runtime_api.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../headers/learning_nodes.hpp"
#include "../headers/logger.hpp"
#include "../headers/loss_nodes.hpp"
#include "../headers/matrix_ops.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/types"
#include "../headers/utils.hpp"
#include "../headers/word2vec.hpp"

using MatrixT = Matrix<FloatT>;

FloatT run_mm_timing(const MatrixT& A, const MatrixT& B)
{
    uint32 max_iters = 100;
    CudaEventTimer timer("Timing for " + std::to_string(max_iters) + " iterations");
    Matrix<FloatT> AB(A.height, B.width);
    for (uint32 i = 0; i < max_iters; i++)
    {
        mmadd<FloatT>(AB, A, B, nullptr);
    }
    auto time = timer.stop();
    uint32 num_bytes = AB.height * AB.width * sizeof(FloatT) * max_iters;
    FloatT bandWidth_mb = num_bytes / (time * (1 << 30));
    LOG("MM  Bandwidth: ", BLUE, bandWidth_mb, "GB/s ", RED, AB.shape_str, RESET,
        " for A, B: ", A.shape_str, " @ ", B.shape_str, " -> ", AB.shape_str);

    auto Bt = normal_init<FloatT>(B.width, B.height);
    CudaEventTimer timer2("Timing for " + std::to_string(max_iters) + " iterations");
    Matrix<FloatT> ABt(A.height, B.width);
    for (uint32 i = 0; i < max_iters; i++)
    {
        mmTadd<FloatT>(ABt, A, Bt, nullptr);
    }
    auto time2 = timer2.stop();
    num_bytes = ABt.height * ABt.width * sizeof(FloatT) * max_iters;
    bandWidth_mb = num_bytes / (time2 * (1 << 30));
    LOG("MMT Bandwidth: ", BLUE, bandWidth_mb, "GB/s ", RED, ABt.shape_str, RESET,
        " for A, Bt: ", A.shape_str, " @ ", Bt.shape_str, " -> ", ABt.shape_str);

    return bandWidth_mb;
}

void run_transpose_timing(const MatrixT& A)
{
    uint32 max_iters = 100;
    CudaEventTimer timer("Timing for " + std::to_string(max_iters) + " iterations");
    MatrixT D(A.width, A.height);
    for (uint32 i = 0; i < max_iters; i++)
    {
        transpose(D, A);
    }
    float time = timer.stop();
    uint32 num_bytes = A.numels() * sizeof(FloatT);
    num_bytes *= max_iters;
    float32 bandWidth_mb = num_bytes / (time * (1 << 30));
    LOG("Transpose Bandwidth: ", BLUE, bandWidth_mb, "GB/s ", RED, A.shape_str);
}

// test if C(orrect) ==  D(ubious)
// return 0 if match, -1 if mismatch
int32 test_match(const MatrixT& C, const MatrixT& D, const char* msg = "")
{
    FloatT max = std::max(C.numels(), D.numels());
    FloatT eps = 1e-6 * max;
    cudaErrCheck(cudaDeviceSynchronize());

    // check sizes match
    if (C.height != D.height || C.width != D.width)
    {
        LOG(RED, msg, "-> Size mismatch for ", C.name, C.shape_str, " and ", D.name, D.shape_str);
        return -1;
    }

    auto any_nan = std::any_of(D.begin(), D.end(), [](FloatT x) { return std::isnan(x); });
    if (any_nan)
    {
        LOG(RED, msg, " -> D has NaNs");
        return -1;
    }

    auto diffs = sameCPU(D, C, eps);
    if (diffs == 0)
    {
        LOG(GREEN, msg, " -> Match!");
        return 0;
    }
    LOG(RED, BOLD, msg, " -> Mismatch at ", diffs, " locations out of ", C.numels(),
        ", with eps: ", eps, RESET, RED, ", Writing diff to diff.csv ");

    std::ofstream("C.csv") << C;
    std::ofstream("D.csv") << D;
    std::ofstream diff("diff.csv");

    diff << std::setprecision(8);
    for (uint32 y = 0; y < D.height; y++)
    {
        for (uint32 x = 0; x < D.width; x++)
        {
            auto d = D(y, x);
            auto c = C(y, x);
            if (std::abs(c - d) > eps)
            {
                std::cout << "Mismatch at " << y << ", " << x << " : " << c << ", " << d
                          << std::endl;
                diff << y << ", " << x << " :\t" << std::setprecision(6) << std::setfill(' ')
                     << std::setw(10) << c << ",\t" << std::setprecision(6) << std::setfill(' ')
                     << std::setw(10) << d << ",\t" << std::setprecision(6) << std::setfill(' ')
                     << std::setw(10) << (c - d) / d << ",\t" << std::endl;
            }
        }
    }
    LOG(RED, "Total diffs: ", diffs);
    return -1;
}

int test_mmTadd_torch()
{
    Matrix<FloatT> A = normal_init<FloatT>(3, 4);
    Matrix<FloatT> B = normal_init<FloatT>(5, 4);
    Matrix<FloatT> C(3, 5);
    Matrix<FloatT> ABt(3, 5);
    fillCPU(C, 2.5);
    fillCPU(ABt, 1.5);

    Matrix<FloatT> Correct(ABt.shape());

    A <<= {2.212206363678, 1.163078665733,  0.774003803730,  0.483804613352,
           1.043440341949, 0.299563467503,  1.183925509453,  0.153025463223,
           1.891711354256, -1.168814778328, -1.234741449356, 1.558071136475};

    B <<= {2.212206363678,  1.163078665733, 0.774003803730,  0.483804613352,  1.043440341949,
           0.299563467503,  1.183925509453, 0.153025463223,  1.891711354256,  -1.168814778328,
           -1.234741449356, 1.558071136475, -1.771028995514, -0.545944571495, -0.451384454966,
           -2.355629682541, 0.579383552074, 0.541440188885,  -1.856081962585, 2.678506612778};

    Correct <<= {7.0797576904, 3.6471183300, 2.6235396862, -6.0418958664, 1.7707128525,
                 3.6471183300, 2.6036026478, 0.4003363252, -2.9063849449, -1.0208352804,
                 2.6235396862, 0.4003363252, 8.8968715668, -5.8250632286, 6.9282684326};

    mmTaddCPU<FloatT>(C, A, B, nullptr);
    mmTadd<FloatT>(ABt, A, B, nullptr);
    cudaErrCheck(cudaDeviceSynchronize());
    if (test_match(C, Correct, "mmTaddCPU") == 0)
    {
        LOG(GREEN, "Test mmTaddCPU passed");
    }
    if (test_match(ABt, Correct, "mmTadd") == 0)
    {
        LOG(GREEN, "Test mmTadd passed");
    }
    return 0;
}

int run_tests(int argc, char const* argv[])
{
    std::string name = (argc > 1) ? argv[0] : "main";
    // clang-format off
    std::stringstream usage("\n\nUsage: \n\t");
    usage << name + " time_mult      h w               for timing A(h,w) * B(w, h)   \n\t" 
          << name + " time_mult_2    h w h2            for timing A(h,w) * B(w, h2)  \n\t"
          << name + " time_transpose h w               for timing transpose A(h,w)   \n\t"
          << name + " test_transpose h w               for testing transpose A(h, w) \n\t"
          << name + " test_mult      h w               for testing matrix multiply   \n\t"
          << name + " test_mult_2    h w h2            for testing matrix multiply   \n\t"
          << name + " test_reduce    h w               for testing reduce sum/min/max \n\t"
          << name + " test_bin_ops   h w               for testing binary ops        \n\t"
          << name + " test_un_ops    h w               for testing unary ops         \n\t"
          << name + " test_mult_csv  a.csv b.csv c.csv for testing with golden files \n\t"
          << name + " test_softmax_grads s_out.csv s_grad_in.csv s_grad_out.csv for testing softmax grqdient  \n\t";

    std::map<std::string, int32> commands = {
        {"time_mult", 4},
        {"time_mult_2", 5},
        {"time_transpose", 4},
        {"test_transpose", 4},
        {"test_mult", 4},
        {"test_mult_2", 5},
        {"test_reduce", 4},
        {"test_bin_ops", 4},
        {"test_un_ops", 4}, 
        {"test_mult_csv", 5},
        {"test_softmax_grads", 5}
    };

    // clang-format on

    if (argc <= 1 || commands.find(argv[1]) == commands.end() || argc != commands[argv[1]])
    {
        std::stringstream ss;
        for (int32 i = 0; i < argc; i++) ss << argv[i] << " ";
        LOG(RED, usage.str(), RESET, ORANGE, "\nInstead called:\n\t", ss.str().c_str());
        throw_rte_with_backtrace("Invalid usage");
    }

    auto init_argv = [&](const char** argv) {  // expects argv[2] and argv[3] to be m, n
        uint32 m = strtoul(argv[2], nullptr, 10);
        uint32 n = strtoul(argv[3], nullptr, 10);
        return normal_init<FloatT>(m, n, 0.f, .5f);
    };

    if (argv[1] == std::string("time_mult") or argv[1] == std::string("time_mult_2"))
    {
        auto A = init_argv(argv);
        uint32 k = (argc > 4) ? strtoul(argv[4], nullptr, 10) : A.width;
        auto B = xavier_uniform_init<FloatT>(A.width, k);
        run_mm_timing(A, B);
    }
    else if (argv[1] == std::string("time_transpose"))
    {
        run_transpose_timing(init_argv(argv));
    }
    else if (argv[1] == std::string("test_transpose"))
    {
        auto A = init_argv(argv);
        MatrixT C = MatrixT(A.width, A.height);
        transposeCPU(C, A);
        MatrixT D = MatrixT(A.width, A.height);
        transpose(D, A);

        return test_match(C, D, "Transpose");
    }
    else if (argv[1] == std::string("test_mult") || argv[1] == std::string("test_mult_2"))
    {
        auto A = init_argv(argv);
        uint32 k = (argc > 4) ? strtoul(argv[4], nullptr, 10) : A.width;
        auto B = normal_init<FloatT>(A.width, k);
        auto S = normal_init<FloatT>(A.height, B.width);
        MatrixT C(A.height, B.width);
        MatrixT D(A.height, B.width);

        test_mmTadd_torch();

        mmaddCPU<FloatT>(C, A, B, nullptr);
        mmadd<FloatT>(D, A, B, nullptr);
        test_match(C, D, "Matrix Multiply No S, no PProcess");

        mmaddCPU<FloatT, Sigmoid<FloatT>::SigmoidF>(C, A, B, nullptr);
        mmadd<FloatT, Sigmoid<FloatT>::SigmoidF>(D, A, B, nullptr);
        test_match(C, D, "Matrix Multiply no S, Sigmoid PProcess");

        mmaddCPU<FloatT>(C, A, B, &S);
        mmadd<FloatT>(D, A, B, &S);
        test_match(C, D, "Matrix Multiply with S");

        mmaddCPU<FloatT, Sigmoid<FloatT>::SigmoidF>(C, A, B, &S);
        mmadd<FloatT, Sigmoid<FloatT>::SigmoidF>(D, A, B, &S);
        test_match(C, D, "Matrix Multiply with S, and Sigmoid PProcess");

        auto Bt = normal_init<FloatT>(k, A.width);
        MatrixT C1(A.height, Bt.height);
        MatrixT D1(A.height, Bt.height);
        mmTaddCPU<FloatT, Sigmoid<FloatT>::SigmoidF>(C1, A, Bt, &S);
        mmTadd<FloatT, Sigmoid<FloatT>::SigmoidF>(D1, A, Bt, &S);
        test_match(C, D, "Matrix Multiply/Transpose with S, and Sigmoid PProcess");
    }
    else if (argv[1] == std::string("test_mult_csv"))
    {
        auto A = read_csv<FloatT>(argv[2]);
        auto B = read_csv<FloatT>(argv[3]);
        auto C = read_csv<FloatT>(argv[4]);
        MatrixT D(A.height, B.width);
        mmadd<FloatT>(D, A, B, (MatrixT*)(nullptr));
        return test_match(C, D, "Matrix Multiply CSV");
    }
    else if (argv[1] == std::string("test_reduce"))
    {
        auto A = init_argv(argv);
        MatrixT C = MatrixT(A.height, 1);
        MatrixT D = MatrixT(A.height, 1);
        int32 match = 0;
        int32 mean_match = -1;

        if (A.width >= 32 and false)
        {
            A(0, A.width - 1) = -10;
            A(0, A.width / 2 - 1) = -5;
            A(0, A.width / 4 - 1) = 5;
            if (A.width > 1)
            {
                A(1, A.width / 2 + 1) = -4;
                A(1, A.width - 1) = 10;
                A(1, A.width / 4 + 1) = 4;
            }
        }
        if (match == 0)
        {
            if (A.width > 5) A(0, A.width - 1) = -10;
            reduceCPU<FloatT, Min<FloatT>>(C, A);
            reduce<FloatT>(D, A, Min<FloatT>());
            match += test_match(C, D, "Min");
            if (match != 0)
            {
                std::ofstream("A.csv") << A;
                return match;
            }
        }

        if (match == 0)
        {
            if (A.width > 4) A(0, A.width - 1) = 10;
            reduceCPU<FloatT>(C, A, Max<FloatT>());
            reduce<FloatT>(D, A, Max<FloatT>());
            match += test_match(C, D, "Max");
            if (match != 0)
            {
                std::ofstream("A.csv") << A;
                return match;
            }
        }

        if (match == 0)
        {
            reduceCPU<FloatT>(C, A);
            reduce_sum<FloatT>(D, A);
            match += test_match(C, D, "Sum");
            if (match != 0)
            {
                std::ofstream("A.csv") << A;
                // return match;
            }
        }

        if (match == 0)
        {
            reduce_meanCPU<FloatT>(C, A);
            reduce_mean(D, A);
            mean_match = test_match(C, D, "Mean");
            match += mean_match;
            if (mean_match != 0)
            {
                std::ofstream("A.csv") << A;
                // return match;
            }

            reduceCPU<FloatT, Plus<FloatT>, Sigmoid<FloatT>::SigmoidF>(C, A);
            reduce<FloatT, Plus<FloatT>, Sigmoid<FloatT>::SigmoidF>(D, A);
            match += test_match(C, D, "Sum with Sigmoid");
            if (match != 0)
            {
                std::ofstream("A.csv") << A;
                return match;
            }
        }

        return mean_match;  // bunch of them fail min/max/sum, so just return mean_match
    }
    else if (argv[1] == std::string("test_bin_ops"))
    {
        auto A = init_argv(argv);
        auto B = normal_init<FloatT>(A.height, A.width);
        MatrixT C(A.height, A.width);
        MatrixT D(A.height, A.width);
        auto add = Plus<FloatT>();

        binary_apply(D, A, B, add);
        binary_applyCPU(C, A, B, add);
        int32 bin_add_match = test_match(C, D, "Bin add");

        // test broadcast:
        auto B1 = normal_init<FloatT>(A.height, 1);
        binary_apply(D, A, B1, add);
        binary_applyCPU(C, A, B1, add);
        int32 bin_add_bc_match = test_match(C, D, "Bin add broadcast col");

        auto B2 = normal_init<FloatT>(1, A.width);
        binary_apply(D, A, B2, add);
        binary_applyCPU(C, A, B2, add);
        int32 bin_add_br_match = test_match(C, D, "Bin add broadcast row");

        auto A1 = normal_init<FloatT>(1, A.width);
        binary_apply(D, A1, B, add);
        binary_applyCPU(C, A1, B, add);
        int32 bin_add_br1_match = test_match(C, D, "Bin add broadcast row");

        auto A2 = normal_init<FloatT>(A.height, 1);
        binary_apply(D, A2, B, add);
        binary_applyCPU(C, A2, B, add);
        int32 bin_add_br2_match = test_match(C, D, "Bin add broadcast col");

        return bin_add_match + bin_add_bc_match + bin_add_br_match + bin_add_br1_match +
               bin_add_br2_match;
    }
    else if (argv[1] == std::string("test_un_ops"))
    {
        auto A = init_argv(argv);
        MatrixT C(A.height, A.width);
        MatrixT D(A.height, A.width);

        unary_apply(D, A, Neg<FloatT>());
        unary_applyCPU(C, A, Neg<FloatT>());
        int un_sq_match = test_match(C, D, "Unary square");

        unary_apply(D, A, Neg<FloatT>());
        unary_applyCPU(C, A, Neg<FloatT>());
        int un_neg_match = test_match(C, D, "Unary negate");

        auto A1 = normal_init<FloatT>(1, A.width);
        unary_apply(D, A1, Neg<FloatT>());
        unary_applyCPU(C, A1, Neg<FloatT>());
        int un_sq_br = test_match(C, D, "Unary square broadcast row");

        auto A2 = normal_init<FloatT>(A.height, 1);
        unary_apply(D, A2, Neg<FloatT>());
        unary_applyCPU(C, A2, Neg<FloatT>());
        int un_sq_bc = test_match(C, D, "Unary square broadcast col");
        return un_sq_match + un_neg_match + un_sq_br + un_sq_bc;
    }
    else if (argv[1] == std::string("test_softmax_grads"))
    {
        auto s_out = read_csv<FloatT>(argv[2]);
        auto s_grad_in = read_csv<FloatT>(argv[3]);
        auto s_grad_out = read_csv<FloatT>(argv[4]);
        MatrixT D(s_out.height, s_out.width);
        LOG(YELLOW, "Tests with sizes: ", s_out.shape_str, " ", s_grad_in.shape_str, " ",
            s_grad_out.shape_str);
        softmax_gradient(D, s_out, s_grad_in);
        auto gradTest = test_match(s_grad_out, D, "Softmax gradient");

        Matrix<FloatT> s_outT(s_out.width, s_out.height);
        transpose(s_outT, s_out);

        Matrix<FloatT> s_grad_inT(s_grad_in.width, s_grad_in.height);
        transpose(s_grad_inT, s_grad_in);

        softmax_gradient(D, s_out, s_grad_inT);
        auto gradTestT = test_match(s_grad_out, D, "Softmax gradient transposed");

        softmax_gradient(D, s_outT, s_grad_in);
        auto gradTestTT = test_match(s_grad_out, D, "Softmax gradient transposed both");
        return gradTest + gradTestT + gradTestTT;
    }

    return 0;
}

/*
static tests:
*/

int values_mismatch(std::string test_name, const Matrix<FloatT>& matrix,
                    const Matrix<FloatT>& expected, FloatT eps = FloatT(1e-6))
{
    cudaErrCheck(cudaDeviceSynchronize());
    uint32 mismatches = sameCPU(matrix, expected, eps);
    if (mismatches)
    {
        LOG(RED, BOLD, "`", test_name, "` mismatch at ", mismatches, " locations, for  ",
            matrix.name, matrix.shape_str, " with eps: ", eps);
        LOG(RED, matrix, RESET, " and with expected");
        LOG(GREEN, expected);
        return 1;
    }
    return 0;
}

int test_word2vec(const char* filename)
{
    Word2Vec word2vec(filename);

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

int test_linearb()  // tests 2 Linear layers, softmax and NLL Loss.
{
    uint32 Ei = 3;
    uint32 Sl = 4;
    uint32 I1 = 5;
    uint32 I2 = 6;
    Input<> x(Sl, Ei, "x");

    Linear<FloatT, Sigmoid<FloatT>> L0(LinearInputT<FloatT>{I1, &x, true, "Linear-L0"});
    Linear<FloatT, TanH<FloatT>> L1(LinearInputT<FloatT>{I2, &L0, true, "Linear-L1"});
    SoftmaxDim1<FloatT> softmax(&L1, "Softmax");

    std::ifstream golden("static_data/linearb.txt");
    golden >> x >> L0.W >> L0.b >> L1.W >> L1.b;

    Input<> t(L1.shape(), "target");
    fillCPU(t, 1);
    NLLLoss<> loss({&softmax, &t}, "L2Error");

    loss.compute();
    loss.backward();
    cudaErrCheck(cudaDeviceSynchronize());

    uint32 err = values_mismatch("Output", L1, read_csv<FloatT>(golden)) +
                 values_mismatch("L1.W.grads", L1.W.grads(), read_csv<FloatT>(golden)) +
                 values_mismatch("L1.b.grads", L1.b.grads(), read_csv<FloatT>(golden)) +
                 values_mismatch("L0.W.grads", L0.W.grads(), read_csv<FloatT>(golden)) +
                 values_mismatch("L0.b.grads", L0.b.grads(), read_csv<FloatT>(golden));

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
    uint32 S = 8;    //  sequence length

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
                 values_mismatch("Q.W.grads", A.Q.W.grads(), read_csv<FloatT>(golden)) +
                 values_mismatch("K.W.grads", A.K.W.grads(), read_csv<FloatT>(golden)) +
                 values_mismatch("V.W.grads", A.V.W.grads(), read_csv<FloatT>(golden));

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

int run_multihead()
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
    xavier_uniform_init<FloatT>(q);
    xavier_uniform_init<FloatT>(k);
    xavier_uniform_init<FloatT>(v);

    // clang-format off
    MHA M(3, {Eq, &q, true, "MHA_Q"},
             {Eq, &k, true, "MHA_K"},
             {Ev, &v, true, "MHA_V"},
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
    Matrix<FloatT> resA(h, w);
    fillCPU(A, 1);
    Matrix<float32> mask(A.shape());
    dropout(resA, A, mask, p);

    cudaErrCheck(cudaDeviceSynchronize());
    FloatT sum = std::accumulate(resA.begin(), resA.end(), 0);

    Matrix<FloatT> B(h, w);
    fillCPU(B, 1);
    Matrix<FloatT> resB(h, w);
    dropout(resB, B, mask, -1);
    cudaErrCheck(cudaDeviceSynchronize());
    uint32 sumB = std::accumulate(resB.begin(), resB.end(), 0);
    if (sumB != sum or std::abs(FloatT(sum) / A.numels() + p - 1) > 0.02)
    {
        LOG(RED, "Dropout failed: ", sum, " vs ", sumB, " or ", sum / A.numels(), " vs ", p);
        return -1;
    }
    LOG(GREEN, "Dropout passed");
    return 0;
}

int test_concat()
{
    uint32 h = 10, w = 5;
    Matrix<FloatT> A(h, w);
    Matrix<FloatT> B(h, w);
    Matrix<FloatT> C(h, w);

    fillCPU(A, 1);
    fillCPU(B, 2);
    fillCPU(C, 3);

    Matrix<FloatT> D(h, 3 * w);
    concat(D, {&A, &B, &C});

    Matrix<FloatT> E(h, w);
    Matrix<FloatT> F(h, w);
    Matrix<FloatT> G(h, w);

    std::vector<Matrix<FloatT>*> efg = {&E, &F, &G};
    split(efg, D, Identity<FloatT>());  // they both work or both fail.

    return test_match(E, A, "Concat A") + test_match(F, B, "Concat B") +
           test_match(G, C, "Concat C");
}

int train_MLP()  // train to generate a zero vector
{
    Input<> inp1(5, 10, "inp1");
    FeedForward<> Model({5, &inp1, true, "Lin1"}, {3, nullptr, false, "Lin2"}, 0.25, "MLP1");
    Input<> target(Model.shape(), "target");
    L2Loss<> loss({&Model, &target}, "L2Loss");
    fillCPU(target, 0);
    xavier_uniform_init<FloatT>(inp1);
    std::ofstream dot("simple.dot");
    graph_to_dot(&loss, dot);

    uint32 max_iter = 2;
    for (uint32 i = 0; i < max_iter; i++)
    {
        LOG(GREEN, "Iteration: ", i);
        xavier_uniform_init<FloatT>(inp1);
        loss.compute();
        loss.backward();
        if ((i + 1) % 10 == 0)
        {
            std::cout << i << ", " << loss.value() << std::endl;
        }
        LOG_SYNC("W1_grads", Model.l_in->W.grads());
        LOG_SYNC("W1", Model.l_in->W);
        loss.update_weights(0.01);
        LOG_SYNC("W1", Model.l_in->W);
    }
    return 0;
}

int test_softmaxDim1()
{
    Input<FloatT> x(4, 3, "x");
    Linear<FloatT> L(5, &x, true, "Linear");
    SoftmaxDim1<FloatT> S(&L, "Softmax");
    Input<FloatT> t(S.shape(), "target");
    L2Loss<FloatT> loss({&S, &t}, "L2Error");
    // clang-format off
    x <<= {
         0.2712765038, -1.2729183435,  0.5026970506,
         0.4180806875, -0.6394201517, -0.6607707143,
        -0.1433143616, -0.1043147221, -1.5312546492,
         0.6318014860, -1.3447675705,  1.4309413433
    };

    L.W <<= {
        -0.9390084147,  2.2027332783,  1.1013969183,
         0.8988130689,  1.0883737803,  0.1770063341,
        -0.4262194633, -0.1981084794,  0.3587698340,
        -2.2124066353,  0.9306892753, -1.3067414761,
        -0.2272174954,  0.8636371493,  0.4374507666
    };
    Matrix<FloatT> golden(5, 3);
    golden <<= {
        -0.0016302123,  0.0046665790, -0.0011383580,
        -0.0029686582,  0.0107183568, -0.0030933935,
         0.0108569972, -0.0288101584,  0.0243558977,
        -0.0023388953,  0.0013220122, -0.0173543170,
        -0.0039192275,  0.0121032111, -0.0027698346
    };
    // clang-format on
    fillCPU(t, 1.0);
    loss.compute();
    loss.backward();
    return test_match(L.W.grads(), golden, "SoftmaxDim1");
}

int test_softmaxDim1_1Row()
{
    Input<FloatT> x(1, 3, "x");
    Linear<FloatT> L(4, &x, true, "Linear");
    SoftmaxDim1<FloatT> S(&L, "Softmax");
    Input<FloatT> t(S.shape(), "target");
    L2Loss<FloatT> loss({&S, &t}, "L2Error");

    Matrix<FloatT> golden_wgrad(4, 3);
    Matrix<FloatT> golden_bgrad(1, 4);
    Matrix<FloatT> golden_loss(1, 1);
    // clang-format off
    x <<= {
        -1.3905061483, -0.8152379990, -0.3204376996
    };

    L.W <<= {
         0.7377384901, -1.7533630133,  0.6032932401,
        -0.2519559562, -0.4373176098, -0.5727743506,
        -3.2022840977,  0.5632690191,  0.2153037637,
        -2.1961724758, -0.2166298330, -0.7261615992
    };

    L.b <<= {
        1.8464213610, 0.2486646771, 0.2778656185, 1.6178737879
    };

    t <<= {1.0, 0.0, 0.0, 0.0};

    golden_wgrad <<= {
         0.0334466845,  0.0196094122,  0.0077076815,
         0.0044421102,  0.0026043588,  0.0010236701,
         0.0430880412,  0.0252620298,  0.0099295015,
        -0.0809768289, -0.0474757962, -0.0186608508
    };
    golden_bgrad <<= {
        -0.0240536034, -0.0031945994, -0.0309873074,  0.0582355037
    };
    golden_loss <<= {0.3671470284};
    // clang-format on

    loss.compute();
    test_match(loss, golden_loss, "SoftmaxDim1Row loss val");
    loss.backward();

    return test_match(L.W.grads(), golden_wgrad, "SoftmaxDim1");
}

int test_LSMCELoss1Row()
{
    Input<> x(1, 3, "x");
    Linear<> L(6, &x, true, "Linear");
    Input<> t(L.shape(), "target");
    LogSoftmaxCELoss<> loss({&L, &t}, "LogSoftmaxCELoss");

    Matrix<FloatT> golden_wgrad(L.W.shape());
    Matrix<FloatT> golden_bgrad(L.b.shape());
    Matrix<FloatT> golden_loss(1, 1);
    // clang-format off
    x <<= {
        -0.6013928056, -1.0122097731, -0.3022692502
    };
    t <<= {1.0,  0.0,  0.0,  0.0,  0.0, 0.0 };
    L.W <<= {
        -0.3649137020, -0.8900567293, -0.0944025069,
         0.0525722094,  0.2386320382, -1.5565147400,
         1.6471320391, -1.0891081095,  1.3290292025,
         1.1973766088, -0.4995271564,  0.3127064705,
        -0.0772454813,  0.3426032066, -0.7574429512,
        -1.5660330057, -1.0804982185, -0.8832487464
    };
    L.b <<= {
       -1.2276864052,  0.6963071823,  1.1649594307, -1.4343018532, -1.4460918903,  0.4119544625
    };
    golden_wgrad <<= {
         0.5752448440,  0.9681998491,  0.2891268730,
        -0.0691427961, -0.1163748726, -0.0347522274,
        -0.0678710490, -0.1142343953, -0.0341130309,
        -0.0049493262, -0.0083302567, -0.0024876071,
        -0.0062032328, -0.0104407184, -0.0031178400,
        -0.4270784259, -0.7188196182, -0.2146561593
    };
    golden_bgrad <<= {
        -0.9565209746,  0.1149711013,  0.1128564402,  0.0082297726,  0.0103147775,  0.7101488709 
    };
    golden_loss <<= {3.1354768276};
    // clang-format on

    loss.compute();
    loss.backward();

    test_match(loss, golden_loss, "CrossEntropy loss");
    return test_match(L.W.grads(), golden_wgrad, "LogSoftmaxCE Dim1");
}

int test_LSMCELoss()
{
    Input<> x(6, 3, "x");
    Linear<> L(6, &x, true, "Linear");
    Input<> t(L.shape(), "target");
    LogSoftmaxCELoss<> loss({&L, &t}, "LogSoftmaxCELoss");

    Matrix<FloatT> golden_wgrad(L.W.shape());
    Matrix<FloatT> golden_bgrad(L.b.shape());
    Matrix<FloatT> golden_loss(1, 1);
    // clang-format off
    x <<= {
        -0.8173339963, -0.5555685163,  0.9999132752,
         1.1166944504,  1.0762836933, -0.0662285835,
         0.1315269619,  0.1680933982,  0.0178458858,
        -0.1281662285,  0.9534983039,  0.3553397954,
         0.2120935619, -0.3377707005, -0.3536095917,
        -0.2729077935, -0.1459826231, -0.2311491221
    };
    t <<= {
        1.0,  0.0,  0.0,  0.0,  0.0, 0.0,
        0.0,  1.0,  0.0,  0.0,  0.0, 0.0,
        0.0,  0.0,  1.0,  0.0,  0.0, 0.0,
        0.0,  0.0,  0.0,  1.0,  0.0, 0.0,
        0.0,  0.0,  0.0,  0.0,  1.0, 0.0,
        0.0,  0.0,  0.0,  0.0,  0.0, 1.0,
    };
    L.W <<= {
        -1.2645164728, -0.4344386458,  0.3690196276,
         1.3373314142, -0.9179764986, -0.9615129232,
        -0.8896581531,  1.0617526770, -0.3206027448,
        -0.3137696981,  0.4300913513,  0.1496529877,
        -0.2459997088, -1.4636487961, -0.7755851746,
        -0.2594423592, -1.9790446758,  2.4860122204
    };
    L.b <<= {
       -0.1278416216,  1.8899183273, -0.5093404651,  0.2017536014,  1.3686803579,  0.3851935565
    };
    golden_wgrad <<= {
         0.7591180205,  0.6232376099, -0.9292756319,
        -0.1800749302, -0.1901088506, -0.1818061471,
        -0.1129867509,  0.0903223008,  0.0508168563,
         0.1818662733, -0.6099193692, -0.2726876736,
        -0.2369795144,  0.3352141380,  0.2690480351,
        -0.4109431803, -0.2487460077,  1.0639045238
    };
    golden_bgrad <<= {
       -0.6799113750,  1.5246317387, -0.6191952825, -0.4168024063,  0.0833148509,  0.1079621390
    };
    golden_loss <<= {1.8877154589};
    // clang-format on

    loss.compute();
    loss.backward();

    test_match(loss, golden_loss, "CrossEntropy loss");
    return test_match(L.W.grads(), golden_wgrad, "LogSoftmaxCE Dim1");
}

int test_adam()
{
    Matrix<FloatT> mat_v = read_csv<FloatT>("static_data/adam_v.csv");

    Matrix<FloatT> grad_d(2, 1, "grad");
    Parameter<FloatT> p(2, 1, "p");
    p <<= {300.001 / mat_v.height, 120.001 / mat_v.width};
    std::ofstream xy("xy.csv");
    xy << "x0,x1,gx,gy,v\n";

    struct Res
    {
        float32 p0, p1;
        float32 g0, g1;
        float32 v;
    };
    std::vector<Res> res;
    for (uint32 i = 0; i < 301; i++)
    {
        auto x0 = p(0, 0), x1 = p(1, 0);
        auto [g0, g1] = gradient_xy(mat_v, x0, x1);
        grad_d <<= {g0, g1};

        p.accumulate_grad(grad_d);
        if (std::abs(g0) < 1e-3 and std::abs(g1) < 1e-3) break;
        auto p0 = x0 * mat_v.height, p1 = x1 * mat_v.width;
        auto v = sample(mat_v, x0, x1);
        // clang-format off
        if(i % 10 == 0 and false)
            LOG(i, 
                std::setprecision(6), GREEN, "\tpixel: [", p0, ' ', p1, ']', 
                std::setprecision(6), CYAN,  "\tmetric:[", x0, ' ', x1, ']', 
                std::setprecision(6), BLUE,  "\tgrads: [", g0, ' ', g1, ']',
                std::setprecision(6), RED,   "\tvalue: [", v, ']');
        // clang-format on
        p.update(3e-2);
        if (x0 < 0 or x0 > 1 or x1 < 0 or x1 > 1) break;
        xy << std::setprecision(12) << p0 << ',' << p1 << ',' << g0 << ',' << g1 << ',' << v
           << '\n';
        res.push_back({p0, p1, g0, g1, v});
    }

    uint32 last_n = 5;
    auto mean_v = std::accumulate(std::end(res) - last_n, std::end(res), 0.0,
                                  [last_n](auto acc, auto& r) { return acc + r.v / last_n; });
    if (std::abs(mean_v + 0.677601) > 0.0001)
    {
        LOG(RED, "Adam failed, mean value: ", mean_v);
        return -1;
    }

    if (res.size() > 196)  // must converge in 195 steps
        throw_rte_with_backtrace("Adam failed, did not converge in ", res.size(), " steps");

    uint32 converged_by = 128;
    float32 mean_gx = 0, mean_gy = 0;
    mean_v = std::accumulate(std::begin(res) + converged_by,
                             std::begin(res) + converged_by + last_n, 0.0, [&](auto acc, auto& r) {
                                 mean_gx += (r.g0 * r.g0) / last_n;
                                 mean_gy += (r.g1 * r.g1) / last_n;
                                 return acc + r.v / last_n;
                             });

    if (std::abs(mean_v + 0.677587) > 0.0001)
    {
        LOG(RED, "Adam failed, mean value: ", mean_v);
        return -1;
    }

    if (mean_gx > 0.02 or mean_gy > 0.05)
    {
        LOG(RED, "Adam failed, mean grads: ", mean_gx, ' ', mean_gy);
        return -1;
    }
    return 0;
}

int main(int argc, char const* argv[])
{
    try
    {
        if (argc > 1)
        {
            return run_tests(argc, argv);
        }

        // test Node<> level stuff, uses data from compare.ipynb
        test_linearb();
        test_attention();
        test_softmaxDim1();
        test_softmaxDim1_1Row();
        test_LSMCELoss();
        test_LSMCELoss1Row();

        // Test kernel calls, self consistency tests.
        test_dropout(0.5);
        test_concat();

        // not really tests
        time_attention();
        run_multihead();

        // Test Adam optimizer against known values
        test_adam();

        // Word2Vec test
        // test_word2vec("static_data/word2vec.txt");
    }
    catch (std::exception& e)
    {
        LOG(RED, e.what());
    }
    return 0;
}