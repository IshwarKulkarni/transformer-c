#include "../headers/nodes.hpp"
#include "../headers/types"
#include <iostream>

int main(int argc, char const* argv[])
{
    Linear<float32> l(3, 4);
    Matrix<float32> x(3, 1);
    l.forward(x);

    std::cout << l.get_output() << std::endl;
    return 0;
}