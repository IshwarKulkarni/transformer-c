#ifndef NODE_PARAMETER_HPP
#define NODE_PARAMETER_HPP

#include <sstream>
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

    Node(uint32_t height, uint32_t width, Node<T>* prev, const std::string& name = "")
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
            snprintf(buffer, 100, "%-20s | %-8s| % 8d | %18s | %5d\n", node->name.c_str(),
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

    virtual std::string repr() { return name + " output: " + output.shape_str; }
};

template <typename TW, typename TG = TW>  // weight and gradient
struct Parameter : Matrix<TW>
{
    void update(float32 lr)
    {
        WeightUpdate<TW, TG> updateFunc(lr);
        binary_apply(updatedWeights, *this, grads, updateFunc);
        std::swap(this->data, updatedWeights.data);
        fill(grads, (TG*)nullptr);
    }
    Parameter(uint32_t height, uint32_t width, TW* wValues = nullptr, std::string name = "Param")
        : Matrix<TW>(normal_init<TW>(height, width)),
          grads(Matrix<TG>(height, width)),
          name(name),
          updatedWeights(height, width)
    {
        fill(grads, (TW*)nullptr);
        fill(*this, wValues);
        fill(updatedWeights, (TW*)nullptr);
    }

 public:
    Matrix<TG> grads;
    const std::string name;

 private:
    Matrix<TW> updatedWeights;
};

template <typename Ta, typename Tb = Ta>
std::ostream& operator<<(std::ostream& os, const Parameter<Ta, Tb>& p)
{
    os << p.name << " Weights: " << p << "\n" << p.name << " Grads: " << p.grads << std::endl;
    return os;
}

#endif  // NODE_PARAMETER_HPP
