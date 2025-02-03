#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include "cuda_runtime_api.h"
#include "datasets.hpp"
#include "functors.cuh"
#include "learning_nodes.hpp"
#include "logger.hpp"
#include "loss_nodes.hpp"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "matrix_ops_cpu.hpp"
#include "nodes.hpp"
#include "types"
#include "utils.hpp"

std::ofstream C_file("C.csv");
std::ofstream D_file("D.csv");
std::ofstream diff_file("diff.csv");

using MatrixT = Matrix<FloatT>;

#define test_match(C, D, msg) \
    test_match_implem(C, D, std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg)

#define test_match_eps(C, D, msg, ef) \
    test_match_implem(C, D, std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg, ef)

int32 test_match_implem(const MatrixT& C, const MatrixT& D, std::string msg, float64 eps = 0.003)
{
    cudaErrCheck(cudaDeviceSynchronize());

    // check sizes match
    if (C.height() != D.height() || C.width() != D.width())
    {
        LOG_NOLOC(RED, msg, "-> Size mismatch for ", C.name, C.shape, " and ", D.name, D.shape);
        return -1;
    }

    auto any_nan = std::any_of(D.begin(), D.end(), [](FloatT x) { return std::isnan(x); });
    if (any_nan)
    {
        LOG_NOLOC(RED, msg, " -> D has NaNs");
        return -1;
    }

    auto is_equal = [eps](const FloatT& a, const FloatT& b) {
        auto ratio = std::abs(a / b - 1);
        if (std::abs(a - b) < 1e-4) return true;
        if (std::abs(a) < 1e-8 and std::abs(b) < 1e-8) return true;  // zeros
        if (std::abs(a) < 1e-4 and std::abs(b) < 1e-4) return ratio < eps * 10;
        return ratio < eps;
    };

    auto diffs = 0;
    for (uint32 i = 0; i < D.numels(); ++i) diffs += (!is_equal(D[i], C[i]));

    if (diffs == 0) return 0;
    LOG_NOLOC(RED, BOLD, msg, " -> Mismatch at ", diffs, " locations out of ", C.numels(),
              ", with eps: ", eps, RESET, RED, ", Writing diff to diff.csv ", RESET, RED,
              "Total diffs: ", diffs, " at eps ", eps);

    diff_file << msg << " total diffs: " << diffs << " of " << C.numels() << " at eps: " << eps
              << "\n";
    diff_file << std::setprecision(8);
    if (C.shape[0] <= 10 and C.shape[1] <= 10 and C.shape[2] <= 5) LOG_NOLOC("C: ", C, "D: ", D);

    diff_file << "b, y, x : \t\t\tC,  \t\t\tD,   \t\t\t|c/d-1|,\t\t c-d\n";
    for (uint32 b = 0; b < D.batch(); b++)
        for (uint32 y = 0; y < D.height(); y++)
            for (uint32 x = 0; x < D.width(); x++)
            {
                // clang-format off
                auto d = D(b, y, x);
                auto c = C(b, y, x);
                if (!is_equal(d, c))
                    diff_file << b << ", " << y << ", " << x << " :\t" 
                              << std::setprecision(6) << std::setfill(' ') << std::setw(14) << c << ",\t"
                              << std::setprecision(6) << std::setfill(' ') << std::setw(14) << d << ",\t" 
                              << std::setprecision(6) << std::setfill(' ') << std::setw(14) 
                              << std::abs(c / d - 1) << ",\t\t" << c - d << std::endl;
                // clang-format off
            }
    diff_file << "\n\n";
    C_file << "\n\n" << msg << " correct, C: " << C;
    D_file << "\n\n" << msg << " computed, D: " << D;

    // if(diffs)
    //    throw_rte_with_backtrace("Mismatch");
    return diffs;
}

int test_mmTadd_torch()
{
    Matrix<FloatT> A = normal_init<FloatT>({3, 4});
    Matrix<FloatT> B = normal_init<FloatT>({5, 4});
    Matrix<FloatT> C({3, 5});
    Matrix<FloatT> ABt({3, 5});
    C.set_val(2.5);
    ABt.set_val(1.5);

    Matrix<FloatT> Correct(ABt.shape);

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

    mmTaddCPU<FloatT>(C, A, B);
    mmTadd<FloatT>(ABt, A, B);
    return test_match(C, Correct, "Multiply with mmTaddCPU") +
           test_match(ABt, Correct, "mmTadd with Torch");
}

template <size_t dim>
int32 test_reduce_implem(const char** argv, bool write_to_files = false)
{
    auto A = init_argv(argv);
    static_assert(dim < 3, "Only 3D tensors supported");

    if (A.shape[dim] == 1)
    {
        LOG("Skipping test for dim ", dim, " as it is already reduced");
        return 0;
    }

    MatrixT C = MatrixT(A.shape.set(dim, 1));
    MatrixT D = MatrixT(A.shape.set(dim, 1));

    std::ofstream Afile, Cfile, Dfile;
    if (write_to_files)
    {
        Afile.open("A.csv", std::ios::app);
        Cfile.open("C.csv", std::ios::app);
        Dfile.open("D.csv", std::ios::app);
    }

    int32 failed = 0;
    if (!failed)
    {
        reduceCPU<FloatT, dim>(C, A);
        reduce<FloatT, dim>(D, A);
        failed += test_match(C, D, "Sums dim_" + std::to_string(dim));
    }

    if (!failed)
    {
        if (A.shape[dim] > 5) A(0, 0, A.width() - 1) = -10;
        using Red = Min<FloatT>;
        reduce<FloatT, dim>(D, A, Red());
        reduceCPU<FloatT, dim, Red>(C, A);
        failed += test_match(C, D, +"Min dim_" + std::to_string(dim));
    }

    if (!failed)
    {
        using Red = Plus<FloatT>;
        reduceCPU<FloatT, dim, Red, Sigmoid<FloatT>::SigmoidF>(C, A);
        reduce<FloatT, dim, Red, Sigmoid<FloatT>::SigmoidF>(D, A);
        failed += test_match(C, D, +"Sum w/ Sigmoid dim_" + std::to_string(dim));
    }

    return failed;
}

int test_reduce_multiple(const char** argv)
{
    auto A = init_argv(argv);
    MatrixT C = shaped_like(A, "C");

    auto init = [&A, &C]() {
        clear_l2_cache();
        normal_init(A, 0, 1);
        C.copy(A.begin());
    };

    uint32 err = 0;

    init();
    if (A.shape[BATCH_IDX] > 1 and A.shape[HEIGHT_IDX] > 1)
    {
        reduce_multiple<FloatT, BATCH_BIT | HEIGHT_BIT>(A);
        reduceCPU<FloatT, BATCH_IDX>(C, C);
        reduceCPU<FloatT, HEIGHT_IDX>(C, C);
        cudaErrCheck(cudaDeviceSynchronize());
        for (uint32 i = 0; i < A.width(); i++)
        {
            auto diff = std::abs(A(0, 0, i) - C(0, 0, i));
            if (std::abs(diff) > 1e-4 * A.shape[0])
            {
                err++;
                LOG(RED, "Mismatch in sum 110, expected: ", C(0, 0, i), " got: ", A(0, 0, i),
                    " vs C: ", C(0, 0, i), ": ", diff);
            }
        }
    }

    init();
    if (A.shape[HEIGHT_IDX] > 1 and A.shape[WIDTH_IDX] > 1)
    {
        reduce_multiple<FloatT, HEIGHT_BIT | WIDTH_BIT>(A);
        reduceCPU<FloatT, WIDTH_IDX>(C, C);
        reduceCPU<FloatT, HEIGHT_IDX>(C, C);

        cudaErrCheck(cudaDeviceSynchronize());
        for (uint32 i = 0; i < A.batch(); ++i)
        {
            auto diff = std::abs(A(i, 0, 0) - C(i, 0, 0));
            if (std::abs(diff) > 1e-3 * A.shape[2])
            {
                err++;
                LOG(RED, "Mismatch in sum 011, expected: ", C(i, 0, 0), " got: ", A(i, 0, 0), ": ",
                    diff);
            }
        }
    }

    init();
    if (A.shape[WIDTH_IDX] > 1 and A.shape[BATCH_IDX] > 1)
    {
        reduce_multiple<FloatT, WIDTH_BIT | BATCH_BIT>(A);
        reduceCPU<FloatT, BATCH_IDX>(C, C);
        reduceCPU<FloatT, WIDTH_IDX>(C, C);
        cudaErrCheck(cudaDeviceSynchronize());
        for (uint32 i = 0; i < A.height(); ++i)
        {
            auto diff = std::abs(A(0, i, 0) - C(0, i, 0));
            if (std::abs(diff) > 1e-4 * A.shape[1])
            {
                err++;
                LOG(RED, "Mismatch in sum 101, expected: ", C(0, i, 0), " got: ", A(0, i, 0),
                    " vs C: ", C(0, i, 0), ": ", diff);
            }
        }
    }

    if (A.shape[WIDTH_IDX] > 1 and A.shape[BATCH_IDX] > 1 and A.shape[HEIGHT_IDX] > 1)
    {
        init();
        reduce_multiple<FloatT, BATCH_BIT | HEIGHT_BIT | WIDTH_BIT>(A);
        reduceCPU<FloatT, BATCH_IDX>(C, C);
        reduceCPU<FloatT, HEIGHT_IDX>(C, C);
        reduceCPU<FloatT, WIDTH_IDX>(C, C);

        if (std::abs(A(0, 0, 0) - C(0, 0, 0)) > 3e-3)
        {
            LOG(RED, "Mismatch in sum 111, expected: ", C(0, 0, 0), " got: ", A(0, 0, 0), " : ",
                std::abs(A(0, 0, 0) - C(0, 0, 0)));
            err++;
        }

        auto color = err ? RED : GREEN;
        LOG(color, err ? "Failed" : "Passed", " reduce multiple");
    }
    return err;
}

int run_parameterized_tests(int argc, char const* argv[])
{
    std::string name = (argc > 1) ? argv[0] : "main";
    // clang-format off
    std::stringstream usage;
    usage << "\n\nUsage: \n\t"
        << name + " test_transpose b h w               for testing transpose A(h, w) \n\t"
        << name + " test_mult      b h w               for testing matrix multiply   \n\t"
        << name + " test_mult_2    b h w h2            for testing matrix multiply   \n\t"
        << name + " test_reduce    b h w               for testing reduce sum/min/max \n\t"
        << name + " test_bin_ops   b h w               for testing binary ops        \n\t"
        << name + " test_un_ops    b h w               for testing unary ops         \n\t"
        << name + " test_mult_csv  a.csv b.csv c.csv for testing with golden files   \n\t";

    std::map<std::string, int32> commands = {
        {"test_transpose", 5},
        {"test_mult", 5},
        {"test_mult_2", 6},
        {"test_reduce", 5},
        {"test_bin_ops", 5},
        {"test_un_ops", 5},
        {"test_mult_csv", 5},
    };

    // clang-format on

    if (argc <= 1 || commands.find(argv[1]) == commands.end() || argc != commands[argv[1]])
    {
        std::stringstream ss;
        ss << RED << usage.str() << RESET << ORANGE << "\nInstead called:\n\t";
        for (int32 i = 0; i < argc; i++) ss << argv[i] << " ";
        ss << RESET << RED;
        if (argc <= 1)
        {
            ss << "\nNo command provided\n";
        }
        if (commands.find(argv[1]) == commands.end())
        {
            ss << "\nInvalid command: " << argv[1] << "\n";
        }
        if (argc != commands[argv[1]])
        {
            ss << "\nInvalid number of arguments: " << argc << " instead of " << commands[argv[1]]
               << "\n";
        }
        LOG(ss.str());
        // throw_rte_with_backtrace("Invalid usage");
        return -2;
    }

    if (argv[1] == std::string("test_transpose"))
    {
        auto A = init_argv(argv);
        MatrixT C = MatrixT(A.shape.t());
        transposeCPU(C, A);
        MatrixT D = MatrixT(A.shape.t());
        transpose(D, A);

        uint32 failed = test_match(C, D, "Transpose");

        transposeCPU(C, A, Sigmoid<FloatT>::SigmoidB());
        MatrixT D1(A.shape.t());
        transpose(D1, A, Sigmoid<FloatT>::SigmoidB());

        failed += test_match(C, D1, "Transpose with Sigmoid");
        return failed;
    }

    else if (argv[1] == std::string("test_mult") || argv[1] == std::string("test_mult_2"))
    {
        auto A = init_argv(argv);
        uint32 k = (argc > 5) ? strtoul(argv[5], nullptr, 10) : A.width();
        auto B = normal_init<FloatT>({A.batch(), A.width(), k});
        auto S = normal_init<FloatT>({A.batch(), A.height(), B.width()});
        MatrixT C = shaped_like(S, "C");
        MatrixT D = shaped_like(S, "D");

        LOG("A.shape: ", A.shape, " B.shape: ", B.shape, " res.shape: ", S.shape);

        uint32 failed = 0;
        auto eps = 1e-4f * C.width();

        mmaddCPU<FloatT>(C, A, B);
        mmadd<FloatT>(D, A, B);
        failed += test_match_eps(C, D, "Matrix Multiply No S, no PProcess", eps);

        mmaddCPU<FloatT, Sigmoid<FloatT>::SigmoidF>(C, A, B);
        mmadd<FloatT, Sigmoid<FloatT>::SigmoidF>(D, A, B);
        failed += test_match_eps(C, D, "Matrix Multiply no S, Sigmoid PProcess", eps);

        mmaddCPU<FloatT>(C, A, B, S);
        mmadd<FloatT>(D, A, B, S);
        failed += test_match_eps(C, D, "Matrix Multiply with S", eps);

        mmaddCPU<FloatT, Sigmoid<FloatT>::SigmoidF>(C, A, B, S);
        mmadd<FloatT, Sigmoid<FloatT>::SigmoidF>(D, A, B, S);
        failed += test_match_eps(C, D, "Matrix Multiply with S, and Sigmoid PProcess", eps);

        test_mmTadd_torch();
        auto Bt = normal_init<FloatT>({A.batch(), k, A.width()});
        MatrixT C1({A.batch(), A.height(), Bt.height()});
        MatrixT D1({A.batch(), A.height(), Bt.height()});
        mmTaddCPU<FloatT, Sigmoid<FloatT>::SigmoidF>(C1, A, Bt, S);
        mmTadd<FloatT, Sigmoid<FloatT>::SigmoidF>(D1, A, Bt, S);
        test_match_eps(C, D, "Matrix Multiply/Transpose with S, and Sigmoid PProcess", eps);
        return failed;
    }
    else if (argv[1] == std::string("test_mult_csv"))
    {
        auto A_ = read_csv<FloatT>(argv[2]);
        auto B_ = read_csv<FloatT>(argv[3]);
        auto C_ = read_csv<FloatT>(argv[4]);

        MatrixT A({&A_, &A_});
        MatrixT B({&B_, &B_});
        MatrixT C({&C_, &C_});

        MatrixT D = shaped_like(C);
        mmadd<FloatT>(D, A, B);
        LOG_SYNC("A.shape: ", A.shape, " B.shape: ", B.shape, " result: ", C.shape);

        auto t = test_match(C, D, "Matrix Multiply CSV");
        return t;
    }
    else if (argv[1] == std::string("test_reduce"))
    {
        // clang-format off
        uint32 failed = 0;
        if (!failed) failed += test_reduce_implem<0>(argv); else return failed;
        if (!failed) failed += test_reduce_implem<1>(argv); else return failed;
        if (!failed) failed += test_reduce_implem<2>(argv); else return failed;
        if (!failed) failed += test_reduce_multiple(argv); else return failed;
        return failed;
        // clang-format on
    }

    else if (argv[1] == std::string("test_bin_ops"))
    {
        auto A = init_argv(argv);
        auto B = normal_init<FloatT>(A.shape, 0.f, 0.5f, "B");
        MatrixT C(A.shape, "C");
        MatrixT D(A.shape, "D");
        auto add = Plus<FloatT>();

        binary_apply(D, A, B, add);
        binary_applyCPU(C, A, B, add);
        int32 bin_add_match = test_match(C, D, "Bin add");

        // test broadcast:
        auto B1 = normal_init<FloatT>({A.batch(), A.height(), 1});
        binary_apply(D, A, B1, add);
        binary_applyCPU(C, A, B1, add);
        int32 bin_add_bc_match = test_match(C, D, "Bin add broadcast col");

        auto B2 = normal_init<FloatT>({A.batch(), 1, A.width()});
        binary_apply(D, A, B2, add);
        binary_applyCPU(C, A, B2, add);
        int32 bin_add_br_match = test_match(C, D, "Bin add broadcast row");

        auto A1 = normal_init<FloatT>({A.batch(), 1, A.width()});
        binary_apply(D, A1, B, add);
        binary_applyCPU(C, A1, B, add);
        int32 bin_add_br1_match = test_match(C, D, "Bin add broadcast row");

        auto A2 = normal_init<FloatT>({A.batch(), A.height(), 1});
        binary_apply(D, A2, B, add);
        binary_applyCPU(C, A2, B, add);
        int32 bin_add_br2_match = test_match(C, D, "Bin add broadcast col");

        return bin_add_match + bin_add_bc_match + bin_add_br_match + bin_add_br1_match +
               bin_add_br2_match;
    }
    else if (argv[1] == std::string("test_un_ops"))
    {
        auto A = init_argv(argv);
        auto C = shaped_like(A);
        auto D = shaped_like(A);

        unary_apply(D, A, Neg<FloatT>());
        unary_applyCPU(C, A, Neg<FloatT>());
        int un_neg_match = test_match(C, D, "Unary negate");

        auto A1 = normal_init<FloatT>({A.batch(), 1, A.width()});
        unary_apply(D, A1, Neg<FloatT>());
        unary_applyCPU(C, A1, Neg<FloatT>());
        int un_neg_br = test_match(C, D, "Unary square broadcast row");

        auto A2 = normal_init<FloatT>({A.batch(), A.height(), 1});
        unary_apply(D, A2, Neg<FloatT>());
        unary_applyCPU(C, A2, Neg<FloatT>());
        int un_neg_bc = test_match(C, D, "Unary square broadcast col");
        if (un_neg_bc == 0) LOG(GREEN, "Passed all unary tests");
        return un_neg_match + un_neg_br + un_neg_bc;
    }
    else if (argv[1] == std::string("test_softmax_grads"))
    {
        auto s_out_ = read_csv<FloatT>(argv[2]);
        auto s_grad_in_ = read_csv<FloatT>(argv[3]);
        auto s_grad_out_ = read_csv<FloatT>(argv[4]);

        Matrix<FloatT> s_out({&s_out_, &s_out_}, "s_out"),
            s_grad_in({&s_grad_in_, &s_grad_in_}, "s_grad_in"),
            s_grad_out({&s_grad_out_, &s_grad_out_}, "s_grad_out");

        MatrixT D(s_grad_out.shape);

        Matrix<FloatT> s_outT(s_out.shape.t());
        transpose(s_outT, s_out);

        Matrix<FloatT> s_grad_inT(s_grad_in.shape.t());
        transpose(s_grad_inT, s_grad_in);

        softmax_gradient(D, s_outT, s_grad_inT);
        return test_match(s_grad_out, D, "Softmax gradient transposed both");
    }

    return 0;
}

/*
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
            LOG(" Option     : ", std::setw(10), options[option], "\tsigma: ", std::setw(5),
            std::setprecision(4), sigma, " Success    : ", GREEN, std::setw(6),
std::setprecision(4), success_rate, "%", RESET, " Avg. calls : ", GREEN, std::setw(6),
std::setprecision(6), nearest_count, RESET, " Avg. time  : ", GREEN, std::setw(6),
std::setprecision(4), timer.stop() * 1e3 / total, "ms.", RESET, " Avg. sim   : ", BLUE,
std::setw(6), std::setprecision(4), meanSim / total, RESET);
            // clang-format on
        }
    }
    // for (auto [word, nearest, sim, nearestSim] : sig033fast)
    //{
    //     LOG(RED, word, " -> ", nearest, BLUE, " error ", sim, "  |nearest-node| = ", nearestSim);
    // }
    return 0;
}
*/

int test_linearb()
{
    uint32 bn, Ei, Sl, I0, I1;
    FloatT expected_loss = 0.0;

    std::ifstream in("static_data/linear.txt");
    in >> bn >> Ei >> Sl >> I0 >> I1 >> expected_loss;

    LOG("TEST LINEARB Batch: ", bn, ", Embedding Size: ", Ei, ", Seq Len: ", Sl,
        ", Lin0 Emb Size: ", I0, ", Lin1 Emb Size: ", I1);

    using act = Sigmoid<FloatT>;
    Input<> x(bn, Sl, Ei, "x");
    Linear<FloatT, act> y1(LinearInputT<FloatT>{I0, &x, false, "y1"});  // yes act, no bias
    Linear<FloatT> y2(LinearInputT<FloatT>{I1, &y1, true, "y2"});       // no act, yes bias

    Input<> t(y2.shape, "target");
    L2Loss<> loss({&y2, &t}, "L2-Error");

    in >> x >> t >> y1.W >> y2.W >> y2.b;

    loss.compute();
    cudaErrCheck(cudaDeviceSynchronize());

    uint32 err = (std::abs(expected_loss - loss.value()) > 1e-5) +
                 test_match_eps(read_csv<FloatT>(in), y2, "Linear1 output", 0.03) +
                 test_match_eps(read_csv<FloatT>(in), y1, "Linear0 output", 0.03);

    if (err)
        throw_rte_with_backtrace("error value mismatch: ", loss.value(), " vs ", expected_loss);

    loss.backward();

    err = test_match(read_csv<FloatT>(in), loss.gradientOut, "loss.gradientOut==y2.grad") +
          test_match(read_csv<FloatT>(in), y2.gradientOut, "y2.gradientOut==y1.grad") +
          test_match(read_csv<FloatT>(in), y2.W.grads(), "y2.W.grads") +
          test_match(read_csv<FloatT>(in), y2.b.grads(), "y2.b.grads") +
          test_match(read_csv<FloatT>(in), y1.W.grads(), "y1.W.grads");
    // test_match(read_csv<FloatT>(golden), y1.b.grads(), "y1.b.grads"); // no bias
    if (err == 0) LOG(GREEN, "Test Linear results match");
    return err;
}

int test_attention()
{
    std::ifstream golden("static_data/attention.txt");
    uint32 bn, Ei, Eq, Ek, Ev, S;
    float32 expected_loss;
    golden >> bn >> Ei >> Eq >> Ek >> Ev >> S >> expected_loss;
    // clang-format off
    Input<> q(bn, S, Ei, "Qi"), 
            k(bn, S, Ei, "Ki"),
            v(bn, S, Ei, "Vi");

    Attention<> A({Eq, &q, false, "Attention_Q"}, 
                  {Ek, &k, false, "Attention_K"},
                  {Ev, &v, false, "Attention_V"}, "Attention");
    
    Input<> target(A.shape, "target");
    L2Loss<> loss({&A, &target}, "L2Error");

    // clang-format on
    golden >> A.Q.W >> A.K.W >> A.V.W >> q >> k >> v >> target;

    loss.compute();
    cudaErrCheck(cudaDeviceSynchronize());

    // if (std::abs(expected_loss - loss.value()) > 1e-5)
    //     throw_rte_with_backtrace("Expected loss values mismatch, expected: ", expected_loss, " vs
    //     ",
    //                              loss.value());

    loss.backward();

    // std::ofstream dot("attention.dot");
    // graph_to_dot(&loss, dot);

    uint32 err = test_match(read_csv<FloatT>(golden), A.qkT, "Attn. qkt") +
                 test_match(read_csv<FloatT>(golden), A.attention_weights, "Attn. smx") +
                 test_match(read_csv<FloatT>(golden), A.attention, "Attn. out") +
                 test_match(read_csv<FloatT>(golden), A.Q, "Query out") +
                 test_match(read_csv<FloatT>(golden), A.K, "Key out") +
                 test_match(read_csv<FloatT>(golden), A.V, "Value out") +
                 test_match(read_csv<FloatT>(golden), A.Q.W.grads(), "Q.W.grads") +
                 test_match(read_csv<FloatT>(golden), A.K.W.grads(), "K.W.grads") +
                 test_match(read_csv<FloatT>(golden), A.V.W.grads(), "V.W.grads");

    if (err == 0) LOG(GREEN, "Test Attention passed");
    return err;
}

int test_productT()
{
    uint32 bn, x0w, Sl, I0, I2, I3;
    FloatT expected_loss = 0.0;

    std::ifstream in("static_data/productT.txt");
    in >> bn >> x0w >> Sl >> I0 >> I2 >> I3 >> expected_loss;

    LOG("TEST PRODUCTT Batch: ", bn, ", x0w: ", x0w, ", Seq Len: ", Sl, ", Lin0 Emb Size: ", I0,
        ", Lin2 Emb Size: ", I2, ", Lin3 Emb Size: ", I3);

    using act = Sigmoid<FloatT>;
    Input<> x0(bn, Sl, x0w, "x0");
    Linear<FloatT, act> y0(LinearInputT<FloatT>{I0, &x0, true, "y1"});

    Input<> x1(bn, I3, I2, "x1");
    Linear<FloatT> y1(LinearInputT<FloatT>{I0, &x1, false, "y2"});

    ProductT<FloatT, DividedBy<FloatT>> A({&y0, &y1}, DividedBy<FloatT>(2.222), "ProductT");
    Input<> t(A.shape, "target");
    L2Loss<> loss({&A, &t}, "L2-Error");

    in >> x0 >> y0.W >> y0.b >> x1 >> y1.W >> t;

    loss.compute();
    cudaErrCheck(cudaDeviceSynchronize());

    uint32 err = (std::abs(expected_loss - loss.value()) > 1e-5);
    if (err) LOG(RED, "error value mismatch: ", loss.value(), " vs ", expected_loss);

    err += test_match_eps(read_csv<FloatT>(in), A, "ProductT output", 0.003) +
           test_match_eps(read_csv<FloatT>(in), y1, "Linear1 output", 0.003) +
           test_match_eps(read_csv<FloatT>(in), y0, "Linear0 output", 0.003);
    if (err) throw_rte_with_backtrace("Outputs mismatch");

    loss.backward();

    err += test_match(read_csv<FloatT>(in), loss.gradientOut, "loss.gradientOut==A.grad") +
           test_match(read_csv<FloatT>(in), y0.W.grads(), "y1.W.grads") +
           test_match(read_csv<FloatT>(in), y0.b.grads(), "y0.b.grads") +
           test_match(read_csv<FloatT>(in), y1.W.grads(), "y1.W.grads");
    if (err == 0) LOG(GREEN, "Test ProductT results match");
    return err;
}

/*
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
*/

int test_LSMCELoss()
{
    Input<> x(3, 5, 3, "x");
    Linear<> L(5, &x, true, "Linear");
    Input<> t(L.shape, "target");
    LogSoftmaxCELoss<> loss({&L, &t}, "LogSoftmaxCELoss");

    float32 expect_loss = 0.0;
    std::ifstream golden("static_data/lsmce.txt");
    golden >> x >> L.W >> L.b >> t >> expect_loss;

    loss.compute();

    auto name = loss.name;
    uint32 err = test_match(read_csv<FloatT>(golden), L, name + "LossVal");
    if (std::abs(expect_loss - loss.value()) > 1e-5)
    {
        LOG(RED, "Loss mismatch: expected ", expect_loss, " vs ", loss.value(),
            " exp/act: ", expect_loss / loss.value());
        err++;
    }

    loss.backward();

    err += test_match(read_csv<FloatT>(golden), loss.gradientOut, name + "loss.gradientOut") +
           test_match(read_csv<FloatT>(golden), L.W.grads(), name + "L.W.grads") +
           test_match(read_csv<FloatT>(golden), L.b.grads(), name + "L.b.grads");
    if (err == 0) LOG(GREEN, "LSMCELoss test passed");
    return err;
}

int test_adam()
{
    Matrix<FloatT> mat_v = read_csv<FloatT>("static_data/adam_v.csv");

    Matrix<FloatT> grad_d({1, 2, 1}, "grad");
    Parameter<FloatT> p({1, 2, 1}, "p");
    p <<= {300.001 / mat_v.height(), 120.001 / mat_v.width()};
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
        auto [g0, g1] = gradient_xy(mat_v, 0, x0, x1);
        grad_d <<= {g0, g1};

        p.accumulate_grad(grad_d);
        if (std::abs(g0) < 1e-3 and std::abs(g1) < 1e-3) break;
        auto p0 = x0 * mat_v.height(), p1 = x1 * mat_v.width();
        auto v = sample(mat_v, 0, x0, x1);
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
    if (std::abs(mean_v + 0.697716) > 0.0001)
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

    if (std::abs(mean_v + 0.697685) > 0.0001)
    {
        LOG(RED, "Adam failed, mean value: ", mean_v);
        return -1;
    }

    if (mean_gx > 0.02 or mean_gy > 0.05)
    {
        LOG(RED, "Adam failed, mean grads: ", mean_gx, ' ', mean_gy);
        return -1;
    }
    LOG(GREEN, "Test Adam passed");
    return 0;
}

int test_dropout(FloatT p)
{
    uint32 b = 1, h = 50, w = 50;
    Matrix<FloatT> A({b, h, w}, "A");
    Matrix<FloatT> resA({b, h, w}, "Res");
    A.set_val(1.f);
    Matrix<float32> mask(A.shape, "mask");
    dropout(resA, A, mask, p);  // generate dropout mask and apply it.

    FloatT sumA = std::accumulate(resA.begin(), resA.end(), 0.f);

    Matrix<FloatT> B({b, h, w});
    B.set_val(1.f);
    Matrix<FloatT> resB({b, h, w}, "ResB");
    dropout(resB, B, mask, -1);  // invalid p, so apply dropput
    cudaErrCheck(cudaDeviceSynchronize());
    auto sumB = std::accumulate(resB.begin(), resB.end(), 0.f);
    if (sumB != sumA)
    {
        LOG(RED, "Dropout failed: ", sumA, " vs ", sumB);
        return -1;
    }

    float32 mask_mean = std::accumulate(mask.begin(), mask.end(), 0.f);
    mask_mean /= mask.numels();

    if (std::abs(mask_mean - 1) > 0.05)  // multiplying by mask should keep matrix sum ~same.
    {
        LOG(RED, "Dropout failed, mask_mean is ", mask_mean, "; too far from 1.");
        return -1;
    }
    LOG(GREEN, "Dropout passed");
    return 0;
}

template <uint32 Dim>
int test_concat_implem()
{
    Shape bhw(20, 40, 300);
    Matrix<FloatT> A(bhw, "A");
    Matrix<FloatT> B(bhw, "B");
    Matrix<FloatT> C(bhw, "C");

    A.set_val(1);
    B.set_val(2);
    C.set_val(3);

    auto dims_merged = bhw.set(Dim, bhw[Dim] * 3);

    Matrix<FloatT> D(dims_merged, "D");
    concat<FloatT, Dim>(D, {&A, &B, &C});

    Matrix<FloatT> E(bhw, "E");
    Matrix<FloatT> F(bhw, "F");
    Matrix<FloatT> G(bhw, "G");

    std::vector<Matrix<FloatT>*> efg = {&E, &F, &G};
    split<FloatT, Dim>(efg, D);  // they both work or both fail.

    return test_match(E, A, "Dim" + std::to_string(Dim) + "_Concat A") +
           test_match(F, B, "Dim" + std::to_string(Dim) + "_Concat B") +
           test_match(G, C, "Dim" + std::to_string(Dim) + "_Concat C");
}

int test_concat()
{
    uint32 err = test_concat_implem<0>() + test_concat_implem<1>() + test_concat_implem<2>();
    if (err == 0) LOG(GREEN, "Concat passed");
    return err;
}

template <typename SoftMaxNode, uint32 SoftmaxDim>
int test_softmaxDim()
{
    static_assert(SoftmaxDim < 2, "Invalid SoftmaxDim");
    uint32 bn, Ei, Sl, I0;
    FloatT expected_loss = 0.0;

    std::ifstream in("static_data/sm_dim" + std::to_string(SoftmaxDim) + ".txt");
    in >> bn >> Ei >> Sl >> I0 >> expected_loss;

    LOG("Batch: ", bn, " Embedding Size: ", Ei, " Seq Len: ", Sl, " Lin Emb Size: ", I0);

    Input<> x(bn, Sl, Ei, "x");
    Linear<FloatT, TanH<FloatT>> L(I0, &x, true, "L");
    SoftMaxNode S(&L);
    Input<FloatT> t(S.shape, "target");
    NLLLoss<FloatT> loss({&S, &t}, "L2Error");  // this is just to test the NLLLoss

    in >> x >> L.W >> L.b >> t;

    loss.compute();

    cudaErrCheck(cudaDeviceSynchronize());
    if (std::abs(expected_loss - loss.value()) > 1e-3)
        LOG(RED, "error value mismatch: ", loss.value(), " vs ", expected_loss);

    uint32 err = test_match_eps(read_csv<FloatT>(in), L, S.name + "linear_output", 1e-4) +
                 test_match_eps(read_csv<FloatT>(in), S, S.name + "output", 1e-4);

    loss.backward();
    if constexpr (SoftmaxDim == 0)
        err += test_match_eps(read_csv<FloatT>(in), S.gradientOutT, S.name + "-y1.grad", 1e-3);
    else
        err += test_match(read_csv<FloatT>(in), S.gradientOut, S.name + "-y1.grad");

    err += test_match(read_csv<FloatT>(in), L.W.grads(), S.name + "-L.W.g");
    err += test_match(read_csv<FloatT>(in), L.b.grads(), S.name + "-L.b.g");

    if (err == 0) LOG(GREEN, S.name, " passed");
    return err;
}

int run_unparameterized_tests()
{
    test_dropout(0.35);
    //  test Node<> level stuff, uses data from compare.ipynb
    test_softmaxDim<SoftmaxDim0<FloatT>, 0>();
    test_softmaxDim<SoftmaxDim1<FloatT>, 1>();

    //  Test kernel calls, self consistency tests.
    test_concat();
    test_attention();
    test_productT();
    test_linearb();
    test_LSMCELoss();

    // Test Adam optimizer against known values
    test_adam();

    /*
    // not really tests
    run_multihead();

    // Word2Vec test
    // test_word2vec("static_data/word2vec.txt");
    */
    return 0;
}

int main(int argc, char const* argv[])
{
    if (argc > 1) return run_parameterized_tests(argc, argv);
    return run_unparameterized_tests();
    return 0;
}