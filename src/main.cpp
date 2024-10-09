#include "../headers/logger.hpp"
#include "../headers/matrix_ops.cuh"
#include "../headers/types"
#include "../headers/utils.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using FloatT = float32;
using MatrixT = Matrix<FloatT>;

float32 run_mm_timing(const MatrixT& A, const MatrixT& B)
{
    uint32 max_iters = 100;
    CudaEventTimer timer("Timing for " + std::to_string(max_iters) + " iterations");
    for (uint32 i = 0; i < max_iters; i++)
    {
        MatrixT D = mmadd<FloatT>(A, B, nullptr);
    }
    float time = timer.stop();
    uint32 num_bytes = A.height * A.height * sizeof(FloatT) * max_iters;
    float32 bandWidth_mb = num_bytes/ (time * (1 << 30));
    LOG("Bandwidth: ", BLUE, bandWidth_mb, "GB/s", RED, A.height, "x", A.height, RESET, " for A",
        A.height, A.width, "x B", B.height, B.width);
    return bandWidth_mb;
}

void run_transpose_timing(const MatrixT& A, const MatrixT& B)
{
    uint32 max_iters = 100;
    CudaEventTimer timer("Timing for " + std::to_string(max_iters) + " iterations");
    for (uint32 i = 0; i < max_iters; i++)
    {
        MatrixT D = transpose(A);
    }
    float time = timer.stop();
    uint32 num_bytes = A.numels() * sizeof(FloatT);
    num_bytes *= max_iters;
    float32 bandWidth_mb = num_bytes/ (time * (1 << 30));
    LOG("Bandwidth: ", BLUE, bandWidth_mb, "GB/s", RED, A.height, "x", A.height, RESET,
        " for AhxAw", A.height, A.width, "x BhBw", B.height, B.width);
}

// test if C(orrect) ==  D(ubious)
int32 test_match(const Matrix<FloatT>& C, const Matrix<FloatT>& D)
{
    FloatT eps = 1e-2;
    cudaErrCheck(cudaDeviceSynchronize());

    // check sizes match
    if (C.height != D.height || C.width != D.width)
    {
        LOG(RED, "-----> Size mismatch for", C.get_name(), "and", D.get_name());
        return -1;
    }

    bool match = same(D, C, eps);
    if (match)
    {
        LOG(GREEN, "-----> Match!");
        return 0;
    }
    LOG(RED, BOLD, "-----> Mismatch!", RESET, RED, "Writing diff to diff.csv");

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
    usage +=
        (name + " time_mult height width                   for time_mult mutliplication  \n\t" +
         name + " time_mult_2 height width height2         for time_mult mutliplication  \n\t" +
         name + " time_transpose height width              for timing mutliplication     \n\t" +
         name + " test_transpose height width              for testing transpose         \n\t" +
         name +
         " test_mult_csv a.csv b.csv c.csv          for testing with folden files \n\t\n"
         "<height2> is optional for second matrix height in multiplication");

    std::map<std::string, uint32> commands = {{"time_mult", 4},
                                              {"time_mult_2", 5},
                                              {"time_transpose", 4},
                                              {"test_transpose", 4},
                                              {"test_mult_csv", 5}};

    if (argc <= 1 || commands.find(argv[1]) == commands.end() || argc != commands[argv[1]])
    {
        std::stringstream ss;
        for (uint32 i = 0; i < argc; i++) ss << argv[i] << " ";
        LOG(RED, usage, RESET, ORANGE, "\nInstead called with ", argc - 1, " args \n\t",
            ss.str().c_str());
        throw std::runtime_error("Invalid usage");
    }

    if (argv[1] == std::string("time_mult") or argv[1] == std::string("time_mult_2"))
    {
        uint32 m = strtoul(argv[2], nullptr, 10);
        uint32 n = strtoul(argv[3], nullptr, 10);
        uint32 k = (argc > 4) ? strtoul(argv[4], nullptr, 10) : n;
        auto A = normal_init<FloatT>(m, n);
        auto B = normal_init<FloatT>(n, k);
        run_mm_timing(A, B);
    }
    else if (argv[1] == std::string("time_transpose"))
    {
        uint32 m = strtoul(argv[2], nullptr, 10);
        uint32 n = strtoul(argv[3], nullptr, 10);
        auto A = normal_init<FloatT>(m, n);
        run_transpose_timing(A, A);
    }
    else if (argv[1] == std::string("test_transpose"))
    {
        uint32 m = strtoul(argv[2], nullptr, 10);
        uint32 n = strtoul(argv[3], nullptr, 10);
        auto A = normal_init<FloatT>(m, n);
        std::ofstream("A.csv") << A;
        auto C = transposeCPU(A);
        auto D = transpose(A);

        return test_match(C, D);
    }
    else if (argv[1] == std::string("test_mult_csv"))
    {
        auto A = read_csv<FloatT>(argv[2]);
        auto B = read_csv<FloatT>(argv[3]);
        auto C = read_csv<FloatT>(argv[4]);
        auto D = mmadd<FloatT>(A, B, nullptr);
        return test_match(C, D);
    }

    return 0;
}
