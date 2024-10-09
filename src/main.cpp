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
    uint32_t max_iter = 5e2;
    // Timer t("--Timing for " + std::to_string(max_iter) + " iterations");
    Timer t("");
    for (int i = 0; i < max_iter; i++)
    {
        MatrixT D = madd<FloatT>(A, B, nullptr);
    }
    uint32 num_bytes = A.height * A.height * sizeof(FloatT) * max_iter;
    float32 bandWidth_mb = num_bytes / (t.get_duration() * (1 << 20));
    LOG("Bandwidth: ", BLUE, bandWidth_mb, " MB/s", RED, A.height, "x", A.height, RESET,
        " for AhxAw", A.height, A.width, "x BhBw", B.height, B.width);
    return bandWidth_mb;
}

void run_transpose_timing(const MatrixT& A, const MatrixT& B)
{
    uint32_t max_iter = 5e2;
    Timer t("--Timing for " + std::to_string(max_iter) + " iterations");
    for (int i = 0; i < max_iter; i++)
    {
        MatrixT D = transpose(A);
    }
    uint32 num_bytes = A.numels() * sizeof(FloatT);
    num_bytes *= max_iter;
    float32 bandWidth_mb = num_bytes / (t.get_duration() * (1 << 20));
    LOG("Bandwidth: ", BLUE, bandWidth_mb, " MB/s");
}

// test if C(orrect) ==  D(ubious)
int32 test_match(const Matrix<FloatT>& C, const Matrix<FloatT>& D)
{
    FloatT eps = 1e-2;
    bool match = same(D, C, eps);
    if (match)
    {
        LOG(GREEN, "-----> Match!");
        return 0;
    }
    LOG(RED, BOLD, "-----> Mismatch!", RESET, RED, "Writing diff to diff.csv");

    std::ofstream diff_file("diff.csv");

    diff_file << std::setprecision(8);
    for (int i = 0; i < D.height; i++)
    {
        for (int j = 0; j < D.width; j++)
        {
            auto d = D(i, j);
            auto c = C(i, j);
            if (std::abs(c - d) > eps)
            {
                diff_file << i << ", " << j << " : " << std::setprecision(8) << std::setw(11) << c
                          << ",\t" << d << std::endl;
            }
        }
    }

    diff_file.close();
    std::ofstream d("D.csv");
    d << D;
    d.close();

    std::ofstream c("C.csv");
    c << C;
    c.close();

    return -1;
}

int main(int argc, char const* argv[])
{
    std::string name = (argc > 1) ? argv[0] : "main";
    std::string usage("\n\nUsage: \n\t");
    usage +=
        (name + " time_mult <m> <n>                   for time_mult mutliplication         \n\t" +
         name + " time_transpose <m> <n>              for timing mutliplication            \n\t" +
         name + " test_transpose <m> <n>              for testing transpose                \n\t" +
         name + " test_mult_csv a.csv b.csv c.csv     for testing multiplication with files\n\t\n");

    std::map<std::string, uint32> commands = {
        {"time_mult", 4}, {"time_transpose", 4}, {"test_transpose", 4}, {"test_mult_csv", 5}};

    if (argc <= 1 || commands.find(argv[1]) == commands.end() || argc != commands[argv[1]])
    {
        std::stringstream ss;
        for (int i = 0; i < argc; i++)
            ss << argv[i] << " ";
        LOG(RED, usage, RESET, ORANGE, "\nInstead called:", argc, "\n\t", ss.str().c_str());

        throw std::runtime_error("Invalid usage");
    }

    if (argv[1] == std::string("time_mult"))
    {
        uint32_t m = strtoul(argv[2], nullptr, 10);
        uint32_t n = strtoul(argv[3], nullptr, 10);
        auto A = normal_init<FloatT>(m, n);
        auto B = normal_init<FloatT>(n, m);
        run_mm_timing(A, B);
    }
    else if (argv[1] == std::string("time_transpose"))
    {
        uint32_t m = strtoul(argv[2], nullptr, 10);
        uint32_t n = strtoul(argv[3], nullptr, 10);
        auto A = normal_init<FloatT>(m, n);
        run_transpose_timing(A, A);
    }
    else if (argv[1] == std::string("test_transpose"))
    {
        uint32_t m = strtoul(argv[2], nullptr, 10);
        uint32_t n = strtoul(argv[3], nullptr, 10);
        auto A = normal_init<FloatT>(m, n);
        auto C = transposeCPU(A);
        auto D = transpose(C);
        return test_match(C, D);
    }
    else if (argv[1] == std::string("test_mult_csv"))
    {
        auto A = read_csv<FloatT>(argv[2]);
        auto B = read_csv<FloatT>(argv[3]);
        auto C = read_csv<FloatT>(argv[4]);
        auto D = madd<FloatT>(A, B, nullptr);
        return test_match(C, D);
    }

    return 0;
}
