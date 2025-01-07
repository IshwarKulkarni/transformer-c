#include <iostream>
#include "matrix.cuh"

int main()
{
    Matrix<float> A({2, 2}, "A");
    std::cout << A;
    return 0;
}