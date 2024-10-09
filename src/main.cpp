#include "../headers/matrix_ops.cuh"
#include "../headers/types"
#include "../headers/utils.hpp"
#include <iostream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

using FloatT = float64;
using MatrixT = Matrix<FloatT>;

void run_timing(const MatrixT& A, const MatrixT& B)
{
    uint32_t max_iter = 5e2;
    Timer t("--Timing with " + std::to_string(max_iter) + " iterations");
    for (int i = 0; i < max_iter; i++)
    {
        MatrixT D = madd<FloatT>(A, B, nullptr);
    }
}

void run_test(const std::string& filename_a, const std::string& filename_b, const std::string& filename_c)
{
    MatrixT A = read_csv<FloatT>(filename_a);
    MatrixT B = read_csv<FloatT>(filename_b);
    MatrixT C = read_csv<FloatT>(filename_c);
    cudaErrCheck(cudaDeviceSynchronize());
    MatrixT D = madd<FloatT>(A, B, nullptr);

    FloatT eps = (sizeof(FloatT) == 4 ? 1e-8 * D.numels() : 1e-5  * D.numels());
    bool match = same(D, C, eps);
    std::ofstream d_file("d.csv");
    if (!match)
    {
        d_file << std::setprecision(8); // <<  D << "\n\n";
        for (int i = 0; i < D.width; i++)
        {
            for (int j = 0; j < D.height; j++)
            {
                auto d = D(i, j);
                auto c = C(i, j);
                if (std::abs(c - d) > eps)
                {
                    d_file << i << " " << j << " " << c << " " << d << std::endl;
                }
            }
        }
    }
    std::cout << (match ? "Match" : "Failed! ") << " at eps: " << eps << std::endl;
}

int main(int argc, char const *argv[])
{

    if(argc == 3)
    {   
        auto A = normal_init<FloatT>(strtoul(argv[1], nullptr, 10),
                                      strtoul(argv[2], nullptr, 10));
        auto B = normal_init<FloatT>(strtoul(argv[2], nullptr, 10),
                                      strtoul(argv[1], nullptr, 10));
        run_timing(A, B);
    }
    else if(argc == 4)
    {
        run_test(argv[1], argv[2], argv[3]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <m> <n> or " << argv[0] << " <a.csv> <b.csv> <c.csv>" << std::endl;
    }
    return 0;
}
