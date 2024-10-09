#include "../headers/logger.hpp"
#include "../headers/matrix_ops.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/types"
#include "../headers/utils.hpp"
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using FloatT = float32;
using MatrixT = Matrix<FloatT>;

FloatT run_mm_timing(const MatrixT& A, const MatrixT& B)
{
    uint32 max_iters = 100;
    CudaEventTimer timer("Timing for " + std::to_string(max_iters) + " iterations");
    Matrix<FloatT> D(A.height, B.width);
    for (uint32 i = 0; i < max_iters; i++)
    {
        mmadd<FloatT, FloatT, FloatT, FloatT>(D, A, B, nullptr);
    }
    auto time = timer.stop();
    uint32 num_bytes = A.height * A.height * sizeof(FloatT) * max_iters;
    FloatT bandWidth_mb = num_bytes / (time * (1 << 30));
    LOG("Bandwidth: ", BLUE, bandWidth_mb, "GB/s ", RED, ": (", A.height, "x", A.height, ")", RESET,
        " for A(", A.height, "x", A.width, ") @ B(", B.height, "x", B.width, ")");
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
    LOG("Bandwidth: ", BLUE, bandWidth_mb, "GB/s ", RED, A.height, "x", A.height, RESET, " for A(",
        A.height, "x", A.width, ")");
}

// test if C(orrect) ==  D(ubious)
int32 test_match(const MatrixT& C, const MatrixT& D, const char* msg = "")
{
    FloatT eps = 1e-2;
    cudaErrCheck(cudaDeviceSynchronize());

    // check sizes match
    if (C.height != D.height || C.width != D.width)
    {
        LOG(RED, msg, "-> Size mismatch for ", C.get_name(), " and ", D.get_name());
        return -1;
    }

    bool match = sameCPU(D, C, eps);
    if (match)
    {
        LOG(GREEN, msg, " -> Match!");
        return 0;
    }
    LOG(RED, BOLD, msg, " -> Mismatch! ", RESET, RED, "Writing diff to diff.csv ");

    std::ofstream("C.csv") << C;
    std::ofstream("D.csv") << D;
    std::ofstream diff("diff.csv");

    diff << std::setprecision(8);
    for (uint32 y = 0; y < D.height; y++)
    {
        for (uint32 x = 0; x < D.width; x++)
        {
            auto d = D(x, y);
            auto c = C(x, y);
            if (std::abs(c - d) > eps)
            {
                diff << y << ", " << x << " :\t" << std::setprecision(6) << std::setfill(' ')
                     << std::setw(10) << c << ",\t" << std::setprecision(6) << std::setfill(' ')
                     << std::setw(10) << d << std::endl;
            }
        }
    }
    return -1;
}

int main(int argc, char const* argv[])
{
    std::string name = (argc > 1) ? argv[0] : "main";
    std::string usage("\n\nUsage: \n\t");
    usage += (name + " time_mult      h w               for timing A(h,w) * B(w, h)    \n\t" +
              name + " time_mult_2    h w h2            for timing A(h,w) * B(w, h2)   \n\t" +
              name + " time_transpose h w               for timing transpose A(h,w)    \n\t" +
              name + " test_transpose h w               for testing transpose A(h, w)  \n\t" +
              name + " test_mult      h w               for testing matrix multiply    \n\t" +
              name + " test_mult_2    h w h2            for testing matrix multiply    \n\t" +
              name + " test_reduce    h w               for testing reduce sum/min/max \n\t" +
              name + " test_bin_ops   h w               for testing binary ops         \n\t" +
              name + " test_un_ops    h w               for testing unary ops          \n\t" +
              name + " test_mult_csv  a.csv b.csv c.csv for testing with golden files  \n\t");

    std::map<std::string, uint32> commands = {
        {"time_mult", 4},    {"time_mult_2", 5}, {"time_transpose", 4}, {"test_transpose", 4},
        {"test_mult", 4},    {"test_mult_2", 5}, {"test_reduce", 4},    {"test_mult_csv", 5},
        {"test_bin_ops", 4}, {"test_un_ops", 4}};

    if (argc <= 1 || commands.find(argv[1]) == commands.end() || argc != commands[argv[1]])
    {
        std::stringstream ss;
        for (uint32 i = 0; i < argc; i++) ss << argv[i] << " ";
        LOG(RED, usage, RESET, ORANGE, "\nInstead called:\n\t", ss.str().c_str());
        throw std::runtime_error("Invalid usage");
    }

    auto normal_init_argv = [&](const char** argv) { // expects argv[2] and argv[3] to be m, n
        uint32 m = strtoul(argv[2], nullptr, 10);
        uint32 n = strtoul(argv[3], nullptr, 10);
        return normal_init<FloatT>(m, n);
    };

    if (argv[1] == std::string("time_mult") or argv[1] == std::string("time_mult_2"))
    {
        auto A = normal_init_argv(argv);
        uint32 k = (argc > 4) ? strtoul(argv[4], nullptr, 10) : A.width;
        auto B = normal_init<FloatT>(A.width, k);
        run_mm_timing(A, B);
    }
    else if (argv[1] == std::string("time_transpose"))
    {
        run_transpose_timing(normal_init_argv(argv));
    }
    else if (argv[1] == std::string("test_transpose"))
    {
        auto A = normal_init_argv(argv);
        MatrixT C = MatrixT(A.width, A.height);
        transposeCPU(C, A);
        MatrixT D = MatrixT(A.width, A.height);
        transpose(D, A);

        return test_match(C, D, "Transpose");
    }
    else if (argv[1] == std::string("test_mult") || argv[1] == std::string("test_mult_2"))
    {
        auto A = normal_init_argv(argv);
        uint32 k = (argc > 4) ? strtoul(argv[4], nullptr, 10) : A.width;
        auto B = normal_init<FloatT>(A.width, k);
        MatrixT C(A.height, B.width);
        MatrixT D(A.height, B.width);
        mmaddCPU<FloatT, FloatT, FloatT, FloatT>(C, A, B, (MatrixT*)(nullptr));
        mmadd<FloatT, FloatT, FloatT, FloatT>(D, A, B, (MatrixT*)(nullptr));
        return test_match(C, D, "Matrix Multiply");
    }
    else if (argv[1] == std::string("test_mult_csv"))
    {
        auto A = read_csv<FloatT>(argv[2]);
        auto B = read_csv<FloatT>(argv[3]);
        auto C = read_csv<FloatT>(argv[4]);
        MatrixT D(A.height, B.width);
        mmadd<FloatT, FloatT, FloatT, FloatT>(D, A, B, (MatrixT*)(nullptr));
        return test_match(C, D, "Matrix Multiply CSV");
    }
    else if (argv[1] == std::string("test_reduce"))
    {
        auto A = normal_init_argv(argv);
        MatrixT C = MatrixT(A.height, 1);
        MatrixT D = MatrixT(A.height, 1);
        reduceCPU<FloatT, Plus<FloatT>>(C, A, 0.f);
        reduce_sum(D, A);
        test_match(C, D, "Sum");

        reduceCPU<FloatT, Max<FloatT>>(C, A, -1e10);
        reduce_max(D, A);
        test_match(C, D, "Max");

        reduceCPU<FloatT, Min<FloatT>>(C, A);
        reduce_min(D, A);
        test_match(C, D, "Min");
    }
    else if (argv[1] == std::string("test_bin_ops"))
    {
        auto A = normal_init_argv(argv);
        auto B = normal_init<FloatT>(A.height, A.width);
        MatrixT C(A.height, A.width);
        MatrixT D(A.height, A.width);
        auto add = Plus<FloatT>();

        binary_apply(D, A, B, add);
        binary_applyCPU(C, A, B, add);
        test_match(C, D, "Bin add");

        // test broadcast:
        auto B1 = normal_init<FloatT>(A.height, 1);
        binary_apply(D, A, B1, add);
        binary_applyCPU(C, A, B1, add);
        test_match(C, D, "Bin add broadcast col");

        auto B2 = normal_init<FloatT>(1, A.width);
        binary_apply(D, A, B2, add);
        binary_applyCPU(C, A, B2, add);
        test_match(C, D, "Bin add broadcast row");

        auto A1 = normal_init<FloatT>(1, A.width);
        binary_apply(D, A1, B, add);
        binary_applyCPU(C, A1, B, add);
        test_match(C, D, "Bin add broadcast row");

        auto A2 = normal_init<FloatT>(A.height, 1);
        binary_apply(D, A2, B, add);
        binary_applyCPU(C, A2, B, add);
        test_match(C, D, "Bin add broadcast col");
    }
    else if (argv[1] == std::string("test_un_ops"))
    {
        auto A = normal_init_argv(argv);
        MatrixT C(A.height, A.width);
        MatrixT D(A.height, A.width);

        unary_apply(D, A, Neg<FloatT>());
        unary_applyCPU(C, A, Neg<FloatT>());
        test_match(C, D, "Unary square");

        unary_apply(D, A, Neg<FloatT>());
        unary_applyCPU(C, A, Neg<FloatT>());
        test_match(C, D, "Unary negate");

        auto A1 = normal_init<FloatT>(1, A.width);
        unary_apply(D, A1, Neg<FloatT>());
        unary_applyCPU(C, A1, Neg<FloatT>());
        test_match(C, D, "Unary square broadcast row");

        auto A2 = normal_init<FloatT>(A.height, 1);
        unary_apply(D, A2, Neg<FloatT>());
        unary_applyCPU(C, A2, Neg<FloatT>());
        test_match(C, D, "Unary square broadcast col");
    }

    return 0;
}
