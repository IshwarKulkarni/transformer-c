#include <sys/types.h>
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "nodes/node.hpp"

static constexpr uint32 det_seed = 42;

uint64 MatrixInitUitls::alloced_bytes = 0;
uint64 MatrixInitUitls::freed_bytes = 0;
uint32 MatrixInitUitls::id = 0;
std::map<uint32, uint64> MatrixInitUitls::id_to_alloced_bytes;

std::random_device rdm::rd;
std::mt19937_64 rdm::det_gen(det_seed);
std::seed_seq rdm::seed({rdm::rd()});
std::mt19937_64 rdm::rdm_gen(seed);
bool rdm::deterministic = true;

uint64 ParameterBase::param_count = 0;

std::vector<NodeBase *> NodeBase::all_nodes;
std::vector<const MatrixBase *> MatrixBase::all_matrices;

template <typename T>
Matrix<T> init_argv(const char **argv, uint32 argc_offset)
{  // expects argv[2..4] be b, h, w;
    uint32 b = strtoul(argv[argc_offset + 0], nullptr, 10);
    uint32 h = strtoul(argv[argc_offset + 1], nullptr, 10);
    uint32 w = strtoul(argv[argc_offset + 2], nullptr, 10);
    return normal_init<T>({b, h, w}, 0.f, 1.f, "A");
};

template <uint32 n_writes>
__global__ void write_dead_beef(uint32 size, uint32 *ptr)
{
    uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
    ptr += idx * n_writes;
    if (idx + n_writes < size)
    {
        for (uint32 i = 0; i < n_writes; ++i)
        {
            ptr[i] = 0xdeadbeef;
        }
    }
}

void clear_l2_cache(uint32 size_bytes)
{
    cudaErrCheck(cudaDeviceSynchronize());
    void *d_data;
    cudaErrCheck(cudaMalloc(&d_data, size_bytes));
    cudaErrCheck(cudaMemset(d_data, 1, size_bytes));

    uint32 n_elems = size_bytes / sizeof(uint32);
    uint32 *u_data = reinterpret_cast<uint32 *>(d_data);

    static constexpr uint32 n_writes = 8;
    dim3 gridDim(iDivUp(n_elems / n_writes, 1024));
    write_dead_beef<n_writes><<<gridDim, 1024>>>(n_elems, u_data);
    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaFree(d_data));
}

template <typename T>
Matrix<T> read_csv(std::ifstream &file)
{
    std::streampos beg = file.tellg();
    if (!file.is_open()) throw_rte_with_backtrace("Could not open file");

    uint32 b = 0, h = 0, w = 0;
    char c;
    file >> c >> b >> h >> w;
    Matrix<FloatT> m({b, h, w});
    file.seekg(beg);
    file >> m;
    return m;
}

template <typename T>
Matrix<T> read_csv(const std::string &filename)
{
    std::ifstream file(filename, std::ios::in);
    return read_csv<T>(file);
}

void write_binary(const Matrix<float32> &m, const std::string &file)
{
    std::ofstream out(file, std::ios::out | std::ios::binary);
    if (!out.is_open()) throw_rte_with_backtrace("Could not open file");

    // write header: byte('#'), byte('F'), byte('4'), 4bytes(b), 4bytes(h), 4bytes(w)
    // write data, 4bytes per element, row-major order, b*h*w elements
    char header[3] = {'#', 'F', '4'};  // magic char, F4 : Float 4 bytes
    out.write(header, 3);
    out.write(reinterpret_cast<const char *>(&m.shape.batch), sizeof(uint32));
    out.write(reinterpret_cast<const char *>(&m.shape.height), sizeof(uint32));
    out.write(reinterpret_cast<const char *>(&m.shape.width), sizeof(uint32));
    out.write(reinterpret_cast<const char *>(m.begin()), m.shape.bytes<float32>());
}

// read binary file written by above write_binary
Matrix<float32> read_binary(const std::string &file = "")
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()) throw_rte_with_backtrace("Could not open file");

    char header[3];
    in.read(header, 3);
    if (header[0] != '#' or header[1] != 'F' or header[2] != '4')
    {
        throw_rte_with_backtrace("Invalid file format");
    }

    uint32 b, h, w;
    in.read(reinterpret_cast<char *>(&b), sizeof(uint32));
    in.read(reinterpret_cast<char *>(&h), sizeof(uint32));
    in.read(reinterpret_cast<char *>(&w), sizeof(uint32));

    std::vector<float32> data(b * h * w);
    in.read(reinterpret_cast<char *>(data.data()), b * h * w * 4);

    Matrix<float32> m({b, h, w});
    m.copy(data.data());
    return m;
}

template <typename T>  // bilinear interpolation, like texture mode border
T bilinear_sample(const Matrix<T> &m, uint32 b, float64 y, float64 x)
{
    if (x < 0 or x >= 1 or y < 0 or y >= 1) return 0;
    uint32 y0 = static_cast<uint32>(y * (m.height() - 1));
    uint32 x0 = static_cast<uint32>(x * (m.width() - 1));
    uint32 y1 = y0 + 1;
    uint32 x1 = x0 + 1;
    if (y1 >= m.height()) y1 = y0;
    if (x1 >= m.width()) x1 = x0;

    float64 y_frac = y * m.height() - y0;
    float64 x_frac = x * m.width() - x0;

    float64 v0 = float64(m(b, y0, x0)) * (1 - y_frac) * (1 - x_frac);
    float64 v1 = float64(m(b, y0, x1)) * (1 - y_frac) * x_frac;
    float64 v2 = float64(m(b, y1, x0)) * y_frac * (1 - x_frac);
    float64 v3 = float64(m(b, y1, x1)) * y_frac * x_frac;
    float64 v(v0 + v1 + v2 + v3);
    return T(v);
}

template <typename T>
T sample(const Matrix<T> &m, uint32 b, float64 y, float64 x, float64 eps)
{
    auto cubicInterpolate = [](const std::array<double, 4> &p, float64 x) {
        return p[1] + 0.5 * x *
                          (p[2] - p[0] +
                           x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
                                x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
    };

    y *= m.height();
    x *= m.width();
    int32 y1 = static_cast<int32>(y);
    int32 x1 = static_cast<int32>(x);

    if (std::abs(y - y1) < eps and std::abs(x - x1) < eps and uint32(x1) < m.width() and
        uint32(y1) < m.height())
    {
        return m(b, y1, x1);
    }

    std::array<std::array<double, 4>, 4> p;

    for (int32 j = -1; j <= 2; ++j)
    {
        for (int32 i = -1; i <= 2; ++i)
        {
            int32 yj = std::max(0, std::min(static_cast<int32>(m.height() - 1), y1 + j));
            int32 xi = std::max(0, std::min(static_cast<int32>(m.width() - 1), x1 + i));
            p[i + 1][j + 1] = m(b, yj, xi);
        }
    }

    std::array<double, 4> arr;
    for (int32 i = 0; i < 4; ++i)
    {
        arr[i] = cubicInterpolate(p[i], y - y1);
    }

    return cubicInterpolate(arr, x - x1);
}

template Matrix<FloatT> init_argv<FloatT>(char const **, unsigned int);

template Matrix<FloatT> read_csv(std::ifstream &file);
template Matrix<FloatT> read_csv(const std::string &filename);

template FloatT sample<FloatT>(Matrix<FloatT> const &, unsigned int, double, double, double);