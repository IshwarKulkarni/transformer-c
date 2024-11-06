#include "../headers/matrix_ops.cuh"

uint32 MatrixIds::alloced_byes = 0;
uint32 MatrixIds::id = 0;

std::random_device rdm::rd;
std::mt19937 rdm::rdm_gen(rd());
std::mt19937 rdm::det_gen(0);
bool rdm::deterministic = false;