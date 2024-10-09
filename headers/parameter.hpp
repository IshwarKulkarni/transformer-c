#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include "matrix.cuh"
#include "matrix_ops.cuh"

template <typename TW, typename TG = TW>  // weight and gradient
struct Parameter
{
    void update(float32 lr)
    {
        WeightUpdate<TW, TG> updateFunc(lr);
        binary_apply(UpdatedWeights, Weights, Grads, updateFunc);
        std::swap(Weights.data, UpdatedWeights.data);
        reset_grad();
    }
    Matrix<TW> Weights;
    Matrix<TG> Grads;
    Matrix<TW> UpdatedWeights;
    const std::string name;

    Parameter(uint32_t height, uint32_t width, TW* wValues = nullptr, std::string name = "Param")
        : Weights(normal_init<TW>(height, width)),
          Grads(Matrix<TG>(height, width)),
          UpdatedWeights(height, width),
          name(name)
    {
        reset_grad();
        if (wValues)
        {
            fill(Weights, wValues);
        }
    }

    inline void reset_grad()
    {
        cudaErrCheck(cudaMemset(Grads.begin(), 0, Grads.numels() * sizeof(TG)));
    }

    void fill_values(TW* wValues) { fill(Weights, wValues); }

    void fill_value(TW value) { fillCPU(Weights, value); }
};

template <typename Ta, typename Tb = Ta>
std::ostream& operator<<(std::ostream& os, const Parameter<Ta, Tb>& p)
{
    os << p.name << " Weights: " << p.Weights << p.name << " Grads: " << p.Grads << std::endl;
    return os;
}

#endif  // PARAMETER_HPP