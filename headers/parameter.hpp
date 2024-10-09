#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include "matrix.cuh"
#include "matrix_ops.cuh"

template <typename T>
struct Node
{
    virtual const Matrix<T>* forward(const Matrix<T>* x) = 0;
    virtual void backward(const Matrix<T>* e) = 0;
    virtual bool update_weights(FloatT lr)
    {
        if (prev) return prev->update_weights(lr);
        return false;
    }

    Node<T>* next = nullptr;  // for now singly linked list, not multiple fan outs.
    Node<T>* prev = nullptr;
    Matrix<T> output;
    const std::string name;

    Node(uint32_t height, uint32_t width, Node<T>* prev, std::string name = "")
        : output(height, width), name(name + "_" + get_layer_num(prev))
    {
        if (prev)
        {
            if (prev->next)
                throw runtime_error_with_backtrace(
                    "Node already has a next node,"
                    "only singly linked nodes are supported");
            prev->next = this;
            this->prev = prev;
        }
    }

    static std::string get_layer_num(Node<T>* node)
    {
        uint32 layer_num = 0;
        while (node)
        {
            node = node->prev;
            layer_num++;
        }
        return std::to_string(layer_num);
    }

    virtual std::string graph_rep(bool traverse_to_init = true)
    {
        char buffer[100];
        std::stringstream ss;
        ss << "Graph " << (prev and !traverse_to_init ? " (does not start from here)\n" : ":\n");
        sprintf(buffer,
                "Layer                |  Output | Param #  |        Shape       | Other params\n");
        ss << buffer << std::string(70, '-') << '\n';
        Node<T>* node = this;
        uint32 n_params = 0;
        uint32 nt_params = 0;
        while (node->prev and traverse_to_init)
        {
            node = node->prev;
        }
        while (node)
        {
            uint32 count = node->n_trainable_params();
            n_params += count;
            uint32 other_params = node->n_untrainable_params();
            nt_params += other_params;
            snprintf(buffer, 100, "%-s | %-8s| % 8d | %18s | %5d\n", node->name.c_str(),
                     node->output.shape_str.c_str(), count, node->params_string().c_str(),
                     node->n_untrainable_params());
            ss << buffer;
            node = node->next;
        }
        ss << std::string(30, '-') << "\n  Total trainable params: " << n_params << '\n'
           << "Total untrainable params: " << nt_params << '\n'
           << std::string(30, '-') << std::endl;
        return ss.str();
    }

    virtual uint32 n_trainable_params() { return 0; }

    virtual uint32 n_untrainable_params() { return 0; }

    virtual std::string params_string() { return ""; }
};

template <typename TW, typename TG = TW>  // weight and gradient
struct Parameter
{
    void update(float32 lr)
    {
        WeightUpdate<TW, TG> updateFunc(lr);
        binary_apply(UpdatedWeights, Weights, Grads, updateFunc);
        std::swap(Weights.data, UpdatedWeights.data);
        set_grad();
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
        set_grad();
        if (wValues)
        {
            fill(Weights, wValues);
        }
    }

    inline void set_grad()
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