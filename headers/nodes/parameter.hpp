/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include "context.hpp"
#include "matrix.cuh"
#include "matrix_ops.hpp"
#include "matrix_ops_cpu.hpp"

struct ParameterBase
{
    virtual ~ParameterBase() = default;

    ParameterBase(Shape s)
    {
        ParameterBase::param_count += s.numels;
        all_params.push_back(this);
    }
    static const std::vector<ParameterBase*>& get_params() { return all_params; }
    uint32 get_update_count() const { return update_count; }
    uint32 get_accum_count() const { return accum_count; }

    static void set_all_is_training(bool is_training)
    {
        for (auto p : all_params) p->is_training = is_training;
    }

    uint64 accum_count = 0;
    uint64 update_count = 0;

    bool is_training = true;

 protected:
 private:
    static uint64 param_count;
    static std::vector<ParameterBase*> all_params;
};

template <typename TW, typename TG = TW>  // weight &&gradient
struct Parameter : Matrix<TW>, public ParameterBase
{
    const float64 beta1 = 0.9;
    const float64 beta2 = 0.999;

    float64 beta1Decayed = 1.0;
    float64 beta2Decayed = 1.0;

    Matrix<TG> m = Matrix<TG>(this->shape, "moment");
    Matrix<TG> v = Matrix<TG>(this->shape, "second_moment");
    Matrix<TG> mag = Matrix<TG>(this->shape, "magnitude");  // temp array for magnitudes

    Parameter(Shape s, std::string name = "Param")
        : Matrix<TW>(xavier_uniform_init<TW>(s.set(2, 1), name)), ParameterBase(s)
    {
        updatedWeights.reset();
        gradients.reset();
        m.reset();
        v.reset();

        LOG_MATRIX_CREATE(" Param", this->name, " : ", this->shape);
    }

    // accumulate the mean of the gradDelta, gradients += gradDelta / batch
    void accumulate_grad(const Matrix<TG>& gradDelta, Context*)
    {
        if (gradDelta.shape.set(BATCH_IDX, 1) != updatedGradients.shape)
        {
            throw_rte_with_backtrace("Shape mismatch for ", this->name, " expected ",
                                     updatedGradients.shape, " but got ", gradDelta.shape);
        }
        if (gradDelta.batch() > 1)
        {
            reduce<TG, BATCH_IDX>(updatedGradients, gradDelta);
            binary_apply(gradients, updatedGradients, Plus<TG>());
        }
        else
        {
            binary_apply(gradients, gradDelta, Plus<TG>());
        }
        LOG_PARAM_UPDATE("Update ", update_count, " Accum ", accum_count, " for ", BLUE, this->name,
                         RESET, " with grad delta shape: ", gradDelta.shape,
                         " &&gradMagnitude: ", RED, grad_magnitude() / this->numels(), RESET);
        accum_count++;
    }

    void update_adam(float32 lr, float32 l1_lambda, float32 l2_lambda, float32 Wfactor)
    // expects gradients to be accumulated in `gradients` and
    // `updatedgradients` to be empty/usable
    {
        /*
        m = beta1 * m + (1.0f - beta1) * gradient;
        v = beta2 * v + (1.0f - beta2) * gradient * gradient;

        // Bias correction
        beta1Decayed *= beta1;
        beta2Decayed *= beta2;
        float mhat = m / (1.0f - beta1Decayed);
        float vhat = v / (1.0f - beta2Decayed);

        // Calculate Adam adjusted gradient
        return alpha * mhat / (std::sqrt(vhat) + epsilon);
        */
        if (accum_count == 0)
        {
            LOG_PARAM_UPDATE(YELLOW, "No gradients accumulated for ", this->name);
            return;
        }

        LOG_PARAM_UPDATE("Updating weights for ", YELLOW, this->name, RESET, " with ", accum_count,
                         " accum'd grads for update# ", update_count, " mag: ", param_magnitude2(),
                         " grad mag: ", grad_magnitude2(), " lr: ", lr);
        if (accum_count > 1)
        {
            unary_apply(gradients, DividedBy<TG>(accum_count));
        }
        if (std::isnan(grad_magnitude2()) && false)
        {
            LOG(RED, "NaN grad magnitude for ", this->name, " with grads: ", gradients);
            exit(0);
        }

        binary_apply(m, gradients, MomentUpdate<TW>(beta1));
        binary_apply(v, gradients, SecondMomentUpdate<TW>(beta2));

        beta1Decayed *= beta1;
        beta2Decayed *= beta2;
        AdamWeightUpdate<TW> awu(beta1Decayed, beta2Decayed);
        binary_apply(updatedGradients, m, v, awu);

        WeightUpdate<TW> wu(lr, Wfactor, l1_lambda, l2_lambda);
        binary_apply(*this, updatedGradients, wu);
    }

    void udate_SGD(float32 lr)
    {
        unary_apply(updatedGradients, gradients, DividedBy<FloatT>(accum_count));
        binary_apply(*this, *this, updatedGradients, WeightUpdate<TW>(lr / accum_count));
        // assign(*(Matrix<TW>*)(this), updatedWeights);
        gradients.reset();
        updatedGradients.reset();
    }

    /* @brief Update the weights using the gradients accumulated so far
     * @param lr: learning rate
     */
    void update(float32 lr, Context*, float32 l1_lambda, float32 l2_lambda, float32 Wfactor)
    {
        update_adam(lr, l1_lambda, l2_lambda, Wfactor);
        accum_count = 0;
        update_count++;
    }

    float64 param_magnitude2() const
    {
        cudaErrCheck(cudaDeviceSynchronize());
        return sqrt(sum_squaredCPU(*this) / this->numels());
    }

    float64 grad_magnitude2() const
    {
        cudaErrCheck(cudaDeviceSynchronize());
        auto mag = sqrt(sum_squaredCPU(updatedGradients) / updatedGradients.numels());
        return mag;
    }

    const Matrix<TG>& grads() const { return gradients; }

    const Matrix<TG>& prev_grads() const { return updatedGradients; }

    // Reset gradients for validation
    void reset_gradients()
    {
        gradients.reset();
        updatedGradients.reset();
    }

    void save_weights(std::ostream& os) const
    {
        uint32 idw = get_type_identifier<TW>();
        uint32 idg = get_type_identifier<TG>();
        uint32 size_type[5] = {this->shape.batch, this->shape.height, this->shape.width, idw, idg};
        uint64 counts[2] = {update_count, accum_count};
        float64 decay[2] = {beta1Decayed, beta2Decayed};

        os.write(reinterpret_cast<const char*>(size_type), sizeof(size_type));
        os.write(reinterpret_cast<const char*>(counts), sizeof(counts));
        os.write(reinterpret_cast<const char*>(decay), sizeof(decay));

        const auto data = this->get_data().get();
        os.write(reinterpret_cast<const char*>(data), this->numels() * sizeof(TW));
    }

    void load_weights(std::istream& is)
    {
        uint32 size_type[5] = {0, 0, 0, 0, 0};
        uint64 counts[2] = {0, 0};
        float64 decay[2] = {0.0, 0.0};

        is.read(reinterpret_cast<char*>(size_type), sizeof(size_type));
        Shape s(size_type[0], size_type[1], size_type[2]);
        if (size_type[3] != get_type_identifier<TW>() || size_type[4] != get_type_identifier<TG>())
            throw_rte_with_backtrace("Type mismatch for ", this->name, " expected ",
                                     get_type_identifier<TW>(), " &&", get_type_identifier<TG>(),
                                     " but got ", size_type[3], " &&", size_type[4]);
        if (s != this->shape)
            throw_rte_with_backtrace("Shape mismatch for ", this->name, " expected ", this->shape,
                                     " but got ", s);

        is.read(reinterpret_cast<char*>(counts), sizeof(counts));
        is.read(reinterpret_cast<char*>(decay), sizeof(decay));

        update_count = counts[0];
        accum_count = counts[1];
        beta1Decayed = decay[0];
        beta2Decayed = decay[1];

        std::vector<TW> data(this->numels());
        is.read(reinterpret_cast<char*>(data.data()), this->numels() * sizeof(TW));
        this->copy(data.data());
    }

 private:
    Matrix<TG> gradients = Matrix<TG>(this->shape, this->name + "grads");
    Matrix<TG> updatedGradients = Matrix<TG>(this->shape, this->name + "updated_grads");
    Matrix<TW> updatedWeights = Matrix<TW>(this->shape, this->name + "updated");
};

template <typename TW = FloatT, typename TG>
std::vector<Parameter<TW, TG>*> get_params_of_type()
{
    std::vector<Parameter<TW, TG>*> params;
    for (auto param : ParameterBase::get_params())
    {
        auto p = dynamic_cast<Parameter<TW, TG>*>(param);
        if (p != nullptr) params.push_back(p);
    }
    if (params.size() != ParameterBase::get_params().size())
        LOG(RED, "There are params of type other than TW=", typeid(TW).name(),
            "/TG=", typeid(TG).name());
    return params;
}

inline void print_param_mags()
{
    for (auto p : get_params_of_type<FloatT, FloatT>())
    {
        auto param_mag = p->param_magnitude2();
        if (param_mag != 0)
            LOG("Magnitude: ", GREEN, param_mag, RESET, "\tGrad Mag: ", RED, p->grad_magnitude2(),
                RESET, "\tfor ", p->name, "\t", p->shape);
    }
}

#endif  // PARAMETER_HPP
