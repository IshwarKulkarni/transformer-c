/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <sys/types.h>
#include "matrix.cuh"
#include "matrix_ops.hpp"
#include "nodes/node.hpp"
#include "types"

static constexpr uint32 det_seed = 0x42;

uint64 MatrixInitUitls::alloced_bytes = 0;
uint64 MatrixInitUitls::freed_bytes = 0;
uint32 MatrixInitUitls::id = 0;
std::map<uint32, uint64> MatrixInitUitls::id_to_alloced_bytes;

std::random_device rdm::rd;
std::mt19937_64 rdm::det_gen(det_seed);
std::seed_seq rdm::seed({rdm::rd()});
std::mt19937_64 rdm::rdm_gen(seed);
bool rdm::deterministic = false;

uint64 ParameterBase::param_count = 0;

std::vector<NodeBase*> NodeBase::all_nodes;
std::vector<const MatrixBase*> MatrixBase::all_matrices;
std::vector<ParameterBase*> ParameterBase::all_params;

template <uint32 n_writes>
__global__ void write_dead_beef(uint32 size, uint32* ptr)
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
    void* d_data;
    cudaErrCheck(cudaMalloc(&d_data, size_bytes));
    cudaErrCheck(cudaMemset(d_data, 1, size_bytes));

    uint32 n_elems = size_bytes / sizeof(uint32);
    uint32* u_data = reinterpret_cast<uint32*>(d_data);

    static constexpr uint32 n_writes = 8;
    dim3 gridDim(iDivUp(n_elems / n_writes, 1024));
    write_dead_beef<n_writes><<<gridDim, 1024>>>(n_elems, u_data);
    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaFree(d_data));
}

template <typename T>
std::pair<cudaTextureObject_t, T*> create_texture_object(const Matrix<T>& in, uint32 batch)
{
    size_t size_bytes = sizeof(T) * in.shape.width;
    size_t height = in.shape.height;
    T* src = in.get_data().get() + batch * in.shape.size2d;
    auto shape = in.shape;

    void* devPtr;
    size_t m_pitch;
    cudaErrCheck(cudaMallocPitch(&devPtr, &m_pitch, size_bytes, height));
    T* data = static_cast<T*>(devPtr);

    cudaErrCheck(
        cudaMemcpy2D(data, m_pitch, src, size_bytes, size_bytes, shape.height, cudaMemcpyDefault));

    // Create texture object
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = data;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
    resDesc.res.pitch2D.width = shape.width;
    resDesc.res.pitch2D.height = shape.height;
    resDesc.res.pitch2D.pitchInBytes = m_pitch;

    cudaTextureDesc texDesc{};
    texDesc.normalizedCoords = 1;  // Use normalized coordinates
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    cudaErrCheck(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    return std::make_pair(texObj, data);
}

__global__ void resample_matrix_kernel(Matrix<float32> out, cudaTextureObject_t texObj,
                                       uint32 batch)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out.shape.width || y >= out.shape.height) return;

    // normalized coordinates
    float32 in_x = static_cast<float32>(x) / (out.shape.width - 1);
    float32 in_y = static_cast<float32>(y) / (out.shape.height - 1);
    out(batch, y, x) = tex2D<float32>(texObj, in_x, in_y);
}

template <typename T>
void resample_matrix(const Matrix<T>& out, Matrix<T>& in)
{
    dim3 block(std::min<uint32>(out.width(), 16), std::min<uint32>(out.height(), 16), 1);
    dim3 grid(out.grid(block));

    cudaErrCheck(cudaGetLastError());

    for (uint32 b = 0; b < out.batch(); b++)
    {
        auto [texObj, data] = create_texture_object(in, b);
        resample_matrix_kernel<<<grid, block>>>(out, texObj, b);
        cudaErrCheck(cudaDeviceSynchronize());
        cudaErrCheck(cudaDestroyTextureObject(texObj));
        cudaErrCheck(cudaFree(data));
    }

    cudaErrCheck(cudaGetLastError());
    cudaErrCheck(cudaDeviceSynchronize());
}

template void resample_matrix(const Matrix<float32>& out, Matrix<float32>& in);

// Structure that holds 5 values &&their corresponding colors
// fill with values > 1 for invalid values
// vals are sorted in ascending order
// colors are in RGBA format
struct AnchorInterpolater
{
    union Color
    {
        uint32 rgba;
        struct
        {
            uint8 a, b, g, r;  // little endian
        };
    };
    float64 vals[5];
    Color colors[5];
};

__global__ void heat_map_kernel(Matrix<uint32> color_image, cudaTextureObject_t texObj, float32 min,
                                float32 max, AnchorInterpolater interp)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= color_image.width() || y >= color_image.height()) return;

    float32 in_x = static_cast<float32>(x) / (color_image.width() - 1);
    float32 in_y = static_cast<float32>(y) / (color_image.height() - 1);
    float32 val = tex2D<float32>(texObj, in_x, in_y);
    val = (val - min) / (max - min);

    if (val < interp.vals[0])
        color_image(0, y, x) = interp.colors[0].rgba;
    else if (val > interp.vals[4])
        color_image(0, y, x) = interp.colors[4].rgba;
    else
    {
        AnchorInterpolater::Color c = {0};
        for (uint32 i = 0; i < 4; i++)
        {
            if (val >= interp.vals[i] && val <= interp.vals[i + 1])
            {
                float64 t = (val - interp.vals[i]) / (interp.vals[i + 1] - interp.vals[i]);
                c.r = static_cast<uint8>(interp.colors[i].r * (1 - t) + interp.colors[i + 1].r * t);
                c.g = static_cast<uint8>(interp.colors[i].g * (1 - t) + interp.colors[i + 1].g * t);
                c.b = static_cast<uint8>(interp.colors[i].b * (1 - t) + interp.colors[i + 1].b * t);
            }
        }
        color_image(0, y, x) = c.rgba;
    }
}

void gen_heat_map(Matrix<uint32>& color_image, const Matrix<float32>& mat_in,
                  const std::string& name)
{
    static std::map<std::string, AnchorInterpolater> interpolaters = {
        {"virdis",
         {.vals = {0, 0.25, 0.5, 0.75, 1},
          .colors = {0x44015400, 0x3b528b00, 0x21918c00, 0x5ec96200, 0xfde72500}}},
        {"inferno",
         {.vals = {0, 0.25, 0.5, 0.75, 1},
          .colors = {0x00000400, 0x57106e00, 0xbc375400, 0xf98e0900, 0xfcffa400}}},
        {"plasma",
         {.vals = {0, 0.25, 0.5, 0.75, 1},
          .colors = {0x0d088700, 0x7e03a800, 0xcc477800, 0xf8954000, 0xf0f92100}}},
        {"twilight",
         {.vals = {0, 0.25, 0.5, 0.75, 1},
          .colors = {0xe1d8e2ff, 0x6175baff, 0x2f1436ff, 0xb25652ff, 0xe1d8e1ff}}}};

    if (mat_in.batch() > 1) throw_rte_with_backtrace("Batch size must be 1");

    if (interpolaters.find(name) == interpolaters.end())
        throw_rte_with_backtrace("Invalid colormap name: ", name);

    dim3 blockDim(std::min<uint32>(mat_in.width(), 16), std::min<uint32>(mat_in.height(), 16), 1);
    dim3 gridDim = color_image.grid(blockDim);

    LOG_MATRIX_OPS("Launching heat_map_kernel with gridDim: ", gridDim, " &&blockDim: ", blockDim,
                   "color_image shape: ", color_image.shape);
    auto [texObj, data] = create_texture_object(mat_in, 0);
    heat_map_kernel<<<gridDim, blockDim>>>(color_image, texObj, -.8, 2, interpolaters[name]);

    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaDestroyTextureObject(texObj));
    cudaErrCheck(cudaFree(data));
}

void write_ppm_image(const Matrix<uint32>& image, std::string name)
{
    std::ofstream file(name);
    file << "P3\n" << image.width() << " " << image.height() << "\n255\n";
    for (uint32 y = 0; y < image.height(); y++)
    {
        for (uint32 x = 0; x < image.width(); x++)
        {
            AnchorInterpolater::Color c = {image(0, y, x)};
            uint32 r = c.r;
            uint32 g = c.g;
            uint32 b = c.b;
            file << r << ' ' << g << ' ' << b << ' ';
        }
        file << '\n';
    }
}
