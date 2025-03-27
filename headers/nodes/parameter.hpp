#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include "matrix.cuh"
#include "matrix_ops.cuh"

struct ParameterBase
{
    virtual ~ParameterBase() = default;

    ParameterBase(Shape s)
    {
        ParameterBase::param_count += s.numels;
        all_params.push_back(this);
    }
    static uint64 get_param_count() { return param_count; }
    static const std::vector<ParameterBase*>& get_all_params() { return all_params; }
    uint32 get_update_count() const { return update_count; }
    uint32 get_accum_count() const { return accum_count; }

 protected:
    uint64 accum_count = 0;
    uint64 update_count = 0;

 private:
    static uint64 param_count;
    static std::vector<ParameterBase*> all_params;
};

template <typename TW, typename TG = TW>  // weight and gradient
struct Parameter : Matrix<TW>, public ParameterBase
{
    const float64 beta1 = 0.9;
    const float64 beta2 = 0.999;

    float64 beta1Decayed = 1.0;
    float64 beta2Decayed = 1.0;

    Matrix<TG> m = Matrix<TG>(this->shape, "moment");
    Matrix<TG> v = Matrix<TG>(this->shape, "second_moment");

    Parameter(Shape s, std::string name = "Param")
        : Matrix<TW>(xavier_uniform_init<TW>(s.set(2, 1), name)), ParameterBase(s)
    {
        updatedWeights.reset();
        gradients.reset();
        m.reset();
        v.reset();

        LOG_TRACE("Parameter ", this->name, this->shape);
    }

    // accumulate the mean of the gradient
    void accumulate_grad(const Matrix<TG>& gradDelta)
    {
        LOG_TRACE("Accumulating gradients for ", BLUE, this->name, RESET,
                  " with grad delta shape: ", gradDelta.shape);
        if (gradDelta.batch() > 1)
        {
            reduce<TG, BATCH_IDX>(updatedGradients, gradDelta);
            binary_apply(gradients, updatedGradients, Plus<TG>());
        }
        else
            binary_apply(gradients, gradDelta, Plus<TG>());
        accum_count++;
    }

    void update_adam(float32 lr)  // expects gradients to be accumulated in `gradients` and
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
            LOG_TRACE(YELLOW, "No gradients accumulated for ", this->name);
            return;
        }

        LOG_TRACE("Updating weights for ", YELLOW, this->name, RESET, " with ", accum_count,
                  " accum'd grads for update# ", update_count);
        if (accum_count > 1)
        {
            unary_apply(gradients, DividedBy<TG>(accum_count));
            accum_count = 0;
        }

        binary_apply(m, gradients, MomentUpdate<TW>(beta1));
        binary_apply(v, gradients, SecondMomentUpdate<TW>(beta2));

        beta1Decayed *= beta1;
        beta2Decayed *= beta2;
        AdamWeightUpdate<TW> awu(beta1Decayed, beta2Decayed);
        binary_apply(updatedGradients, m, v, awu);

        binary_apply(*this, updatedGradients, WeightUpdate<TW>(lr));

        gradients.reset();
    }

    void udate_SGD(float32 lr)
    {
        unary_apply(updatedGradients, gradients, DividedBy<FloatT>(accum_count));
        binary_apply(updatedWeights, *this, updatedGradients, WeightUpdate<TW>(lr / accum_count));
        assign(*(Matrix<TW>*)(this), updatedWeights);
        fill(gradients, (TG*)nullptr);
        accum_count = 0;
    }

    /* @brief Update the weights using the gradients accumulated so far
     * @param lr: learning rate
     */
    void update(float32 lr)
    {
        // udate_SGD(lr);
        update_adam(lr);
        update_count++;
    }

    float64 param_magnitude() const
    {
        cudaErrCheck(cudaDeviceSynchronize());
        return sqrt(sum_squaredCPU(*this) / this->numels());
    }

    float64 grad_magnitude() const
    {
        cudaErrCheck(cudaDeviceSynchronize());
        return sqrt(sum_squaredCPU(gradients) / gradients.numels());
    }

    const Matrix<TG>& grads() const { return gradients; }

    void set_is_training(bool is_training) { this->is_training = is_training; }

    void save_weights(std::ostream& os) const
    {
        uint32 id = get_type_identifier<TW>();
        uint32 size_type[4] = {this->shape.batch, this->shape.height, this->shape.width, id};
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
        uint32 size_type[4] = {0, 0, 0, 0};
        uint64 counts[2] = {0, 0};
        float64 decay[2] = {0.0, 0.0};

        is.read(reinterpret_cast<char*>(size_type), sizeof(size_type));
        Shape s(size_type[0], size_type[1], size_type[2]);
        if (size_type[3] != get_type_identifier<TW>())
            throw_rte_with_backtrace("Type mismatch for ", this->name, " expected ",
                                     get_type_identifier<TW>(), " but got ", size_type[3]);
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
    bool is_training = true;
    Matrix<TG> gradients = Matrix<TG>(this->shape, this->name + "grads");
    Matrix<TG> updatedGradients = Matrix<TG>(this->shape, this->name + "updated_grads");
    Matrix<TW> updatedWeights = Matrix<TW>(this->shape, this->name + "updated");
};

#endif  // PARAMETER_HPP
