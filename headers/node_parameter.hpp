#ifndef NODE_PARAMETER_HPP
#define NODE_PARAMETER_HPP

#include "functors.cuh"
#include "logger.hpp"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "matrix_ops_cpu.hpp"
#include "types"

template <typename TW, typename TG = TW>  // weight and gradient
struct Parameter : Matrix<TW>
{
    const float64 beta1 = 0.9;
    const float64 beta2 = 0.999;

    float64 beta1Decayed = 1.;
    float64 beta2Decayed = 1.;

    Matrix<TG> m = Matrix<TG>(this->shape, "moment");
    Matrix<TG> v = Matrix<TG>(this->shape, "second_moment");
    Matrix<TG> m_updated = Matrix<TG>(this->shape, "moment_updated");
    Matrix<TG> v_updated = Matrix<TG>(this->shape, "second_moment_updated");

    Parameter(Shape s, std::string name = "Param")
        : Matrix<TW>(xavier_uniform_init<TW>(s.set(2, 1), name))
    {
        updatedWeights.reset();
        gradients.reset();
        m.reset();
        v.reset();
    }

    void assign(Matrix<TW>& a, Matrix<TW>& b)
    {
        cudaMemcpy(a.get().get(), b.get().get(), b.numels() * sizeof(TW), cudaMemcpyDefault);
        b.reset();
    }

    // accumulate the mean of the gradient
    void accumulate_grad(const Matrix<TG>& gradDelta)
    {
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

        bool ct_between1 = (update_count < 10);
        bool ct_between2 = (update_count < 10);

        bool debug =
            ((this->height() == 8 and ct_between1) or (this->height() == 3 and ct_between2));
        auto mag = [](const Matrix<TW>& x) {
            cudaErrCheck(cudaDeviceSynchronize());
            return sqrt(sum_squaredCPU(x) / x.numels());
        };

        if (accum_count == 0)
        {
            LOG(ORANGE, "No gradients accumulated, skipping update");
            return;
        }

        unary_apply(updatedGradients, gradients, DividedBy<TG>(accum_count));
        assign(gradients, updatedGradients);
        accum_count = 0;

        binary_apply(m_updated, m, gradients, MomentUpdate<TW>(beta1));
        binary_apply(v_updated, v, gradients, SecondMomentUpdate<TW>(beta2));
        if (debug)
        {
            LOG_SYNC(YELLOW, update_count, "> beta1Decayed: ", beta1Decayed, " beta2decayed ",
                     beta2Decayed, " gradients mag: ", mag(gradients), RESET, " ", *this, m_updated,
                     v_updated);
        }

        beta1Decayed *= beta1;
        beta2Decayed *= beta2;
        AdamWeightUpdate<TW> awu(beta1Decayed, beta2Decayed);
        binary_apply(updatedGradients, m_updated, v_updated, awu);

        if (debug)
        {
            LOG_SYNC(RED, "After AWU mag of update ", mag(updatedGradients), RESET,
                     updatedGradients);
        }

        binary_apply(updatedWeights, *this, updatedGradients, WeightUpdate<TW>(lr));
        assign(*(Matrix<TW>*)(this), updatedWeights);
        if (debug) LOG_SYNC("This after update ", *this);

        assign(m, m_updated);
        assign(v, v_updated);
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
        return sqrt(sum_absCPU(*this) / (accum_count * this->numels));
    }

    float64 grad_magnitude() const
    {
        cudaErrCheck(cudaDeviceSynchronize());
        return sqrt(sum_absCPU(gradients) / (accum_count * gradients.numels));
    }

    const Matrix<TG>& grads() const { return gradients; }

    void set_is_training(bool is_training) { this->is_training = is_training; }

 private:
    uint32 update_count = 0;
    uint32 accum_count = 0;
    bool is_training = true;
    Matrix<TG> gradients = Matrix<TG>(this->shape, this->name + "grads");
    Matrix<TG> updatedGradients = Matrix<TG>(this->shape, this->name + "updated_grads");
    Matrix<TW> updatedWeights = Matrix<TW>(this->shape, this->name + "updated");
};

template <typename T>
struct Node;
template <typename T = FloatT>
using NodePtr = Node<T>*;

template <typename T = FloatT>
using NodePtrs = std::vector<NodePtr<T>>;

template <typename T = FloatT>
struct Node : Matrix<T>
{
    Node(Shape s, const NodePtrs<T>& prevs, const std::string& name_, uint32 prev_count)
        : Matrix<T>(s, name_)
    {
        if (prevs.size() != prev_count)
            throw_rte_with_backtrace("Expected ", prev_count, " input(s), for ", name_, " got ",
                                     prev_nodes.size());
        for (auto& p : prevs)
        {
            if (p == nullptr) throw_rte_with_backtrace("Input node is nullptr");
            prev_nodes.push_back(p);
        }
    }

    // call compute on all previous nodes to populate their outputs, then call forward
    virtual void compute(uint32 depth = 0)
    {
        for (auto& p : prev_nodes)
        {
            p->compute(depth + 1);
        }
        this->forward();
    }
    virtual void forward() = 0;
    virtual void backward(const Matrix<T>* e) = 0;
    virtual void update_weights(FloatT lr)
    {
        for (auto& p : params) p->update(lr);
        for (auto& n : prev_nodes)
        {
            n->update_weights(lr);
        }
    }

    std::vector<Parameter<T, T>*> params;
    NodePtrs<T> prev_nodes{};

    Matrix<T>& prev(uint32 i) { return *((Matrix<T>*)(prev_nodes[i])); }

    virtual uint32 param_count()
    {
        uint32 total = 0;
        for (auto& p : params) total += p->numels();
        return total;
    }
    virtual std::string dot_repr() { return " [label=\"" + this->name + "\" shape=rect]\n"; }

    bool is_training = true;

    void set_is_training(bool is_training)
    {
        this->is_training = is_training;
        for (auto& p : params) p->set_is_training(is_training);
        for (auto& p : prev_nodes) p->set_is_training(is_training);
    }
};

template <typename T = FloatT>
void graph_to_dot(NodePtr<T> node, std::ostream& os, std::string header = "digraph G")
{
    NodePtrs<T> nodes;
    nodes.push_back(node);
    std::set<std::string> edge_strs;
    std::set<std::string> node_strs;
    char edge_buffer[256];
    while (!nodes.empty())
    {
        auto* n = nodes.back();
        nodes.pop_back();
        for (auto* p : n->prev_nodes)
        {
            std::string shape = p->shape;
            snprintf(edge_buffer, 256, "%d -> %d [label=\"%s\" spline=ortho ]\n", p->id, n->id,
                     shape.c_str());
            edge_strs.insert(edge_buffer);
            nodes.push_back(p);
        }
        node_strs.insert(std::to_string(n->id) + n->dot_repr());
    }
    os << header << "{\n"
       << "compound=true;\n";
    for (const auto& edge : edge_strs) os << edge << "\n";
    for (const auto& node_str : node_strs) os << node_str << "\n";
    os << "}\n" << std::endl;
}

template <typename Ta, typename Tb = Ta>
std::ostream& operator<<(std::ostream& os, const Parameter<Ta, Tb>& p)
{
    os << p.name << " Weights: " << *(Matrix<Ta>*)(&p) << " With Grads: " << p.grads() << std::endl;
    return os;
}

#endif  // NODE_PARAMETER_HPP
