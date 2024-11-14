#include <cuda_device_runtime_api.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include "../headers/logger.hpp"
#include "../headers/matrix_ops.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/types"
#include "../headers/utils.hpp"

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

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
    FloatT eps = 1e-2 * max;
    cudaErrCheck(cudaDeviceSynchronize());

    // check sizes match
    if (C.height != D.height || C.width != D.width)
    {
        LOG(RED, msg, "-> Size mismatch for ", C.name, " and ", D.name);
        return -1;
    }

    bool match = sameCPU(D, C, eps);
    if (match)
    {
        LOG(GREEN, msg, " -> Match!");
        return 0;
    }
    LOG(RED, BOLD, msg, " -> Mismatch with eps: ", eps, RESET, RED, ", Writing diff to diff.csv ");

    std::ofstream("C.csv") << C;
    std::ofstream("D.csv") << D;
    std::ofstream diff("diff.csv");

    diff << std::setprecision(8);
    uint32 diffs = 0;
    for (uint32 y = 0; y < D.height; y++)
    {
        for (uint32 x = 0; x < D.width; x++)
        {
            auto d = D(y, x);
            auto c = C(y, x);
            if (std::abs(c - d) > eps)
            {
                diffs++;
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

int main(int argc, char const* argv[])
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
        auto B = xavier_init<FloatT>(A.width, k);
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
