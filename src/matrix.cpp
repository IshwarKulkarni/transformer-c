#include "../headers/matrix_ops.cuh"

static constexpr uint32 det_seed = 42;

uint32 MatrixInitUitls::alloced_byes = 0;
uint32 MatrixInitUitls::id = 0;

std::random_device rdm::rd;
std::mt19937 rdm::det_gen(det_seed);
std::seed_seq rdm::seed({rdm::rd(), rdm::rd(), rdm::rd(), rdm::rd(), rdm::rd()});
std::mt19937 rdm::rdm_gen(seed);
bool rdm::deterministic = true;