#include "../headers/matrix_ops.cuh"
#include "../headers/types"
#include <iostream>
#include <iostream>
#include <fstream>
#include <iterator>
#include <type_traits>
#include <vector>
#include <iomanip>
#include <chrono>

#include <typeinfo>

using FloatT = float32;
using MatrixT = Matrix<FloatT>;

MatrixT read_csv(const std::string& filename)
{
    std::ifstream file(filename, std::ios::in);
    uint32_t m, n;
    file >> m >> n;
    std::vector<FloatT> data(m * n);
    std::copy(std::istream_iterator<float>(file), std::istream_iterator<float>(), data.begin());
    MatrixT matrix(m, n, data.data());
    return matrix;
}

struct Timer
{
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::string name;
    Timer(const char* name) : name(name)
    {
        t1 = std::chrono::high_resolution_clock::now();
    }
    ~Timer()
    {
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << name << " Time: " << time_span.count() << " seconds." << std::endl;
    }
};

void run_timing(const MatrixT& A, const MatrixT& B)
{
    Timer t("1000 madd");
    for (int i = 0; i < 1000; i++)
    {
        MatrixT D = madd<FloatT>(A, B, nullptr);
    }
}

void run_test(const MatrixT &D, const MatrixT &C)
{
    FloatT eps = (sizeof(FloatT) == 4 ? 1e-8 * D.numels() : 1e-5  * D.numels());
    bool match = same(D, C, eps);
    std::ofstream d_file("d.csv");
    if (!match)
    {
        // print where D is different from C
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

int main()
{
    MatrixT A = read_csv("data/a.csv");
    MatrixT B = read_csv("data/b.csv");
    MatrixT C = read_csv("data/c.csv");
    cudaErrCheck(cudaDeviceSynchronize());

    std::cout << " Matrix A: " << A.get_name() << std::endl
              << " Matrix B: " << B.get_name() << std::endl
              << " Matrix C: " << C.get_name() << std::endl;  

    MatrixT D = madd<FloatT>(A, B, nullptr);

    run_test(D, C);
    //run_timing(A, B);
    return 0;
}

