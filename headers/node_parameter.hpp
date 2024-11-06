#ifndef NODE_PARAMETER_HPP
#define NODE_PARAMETER_HPP

#include "matrix.cuh"
#include "matrix_ops.cuh"

template <typename TW, typename TG>
struct Parameter;

template <typename T = FloatT>
struct Node : Matrix<T>
{
    const uint32 id = node_id();
    Node(uint32_t height, uint32_t width, const std::vector<Node<T>*>& prev,
         const std::string& name, uint32 prev_count)
        : Matrix<T>(height, width), name(name + '{' + std::to_string(id) + '}')
    {
        if (prev.size() != prev_count)
            throw_rte_with_backtrace("Expected ", prev_count, " input(s), for ", name, " got ",
                                     prev_nodes.size());
        for (auto& p : prev)
        {
            if (p == nullptr) throw_rte_with_backtrace("Input node is nullptr");
            prev_nodes.push_back(p);
            p->next_nodes.push_back(this);
        }
    }

    Node(std::pair<uint32_t, uint32_t> shape, const std::vector<Node<T>*>& prev,
         const std::string& name, uint32 prev_count)
        : Node(shape.first, shape.second, prev, name, prev_count)
    {
    }

    virtual void forward(uint32 depth = 0)
    {
        for (auto& p : prev_nodes)
        {
            p->forward(depth + 1);
        }
        this->compute();
    }
    virtual void compute() = 0;
    virtual void backward(const Matrix<T>* e) = 0;
    virtual void update_weights(FloatT lr)
    {
        for (auto& p : params) p->update(lr);
        for (auto& n : next_nodes)
        {
            n->update_weights(lr);
        }
    }

    std::vector<Parameter<T, T>*> params;
    std::vector<Node<T>*> next_nodes{};
    std::vector<Node<T>*> prev_nodes{};
    const std::string name;

    Matrix<T>& prev(uint32 i) { return *((Matrix<T>*)(prev_nodes[i])); }

    Node<T>* prev_node(uint32 i) { return prev_nodes[i]; }

    virtual uint32 n_trainable_params()
    {
        uint32 total = 0;
        for (auto& p : params) total += p->numels();
        return total;
    }

    virtual uint32 n_untrainable_params() { return 0; }

    virtual std::string dot_repr() { return ""; }

 private:
    uint32 node_id()
    {
        static uint32 id = 0;
        return id++;
    }
};

template <typename T = FloatT>
void graph_to_dot(Node<T>* node, std::ostream& os, const std::string& header = "digraph G")
{
    os << header << " {\n";
    std::vector<Node<T>*> nodes;
    nodes.push_back(node);
    while (!nodes.empty())
    {
        auto* n = nodes.back();
        nodes.pop_back();
        for (auto* p : n->prev_nodes)
        {
            os << p->id << "->" << n->id << " [label=\" " << p->shape_str << " \"]\n";
            nodes.push_back(p);
        }
        os << n->id << " [label=\"" << n->name << "\"]\n" << n->dot_repr();
    }
    os << "}\n";
}

template <typename T = FloatT>
using NodePtrs = const std::vector<Node<T>*>&;

template <typename T = FloatT>
struct Input : Node<T>
{
    Input(uint32_t height, uint32_t width, const std::string& name = "Input")
        : Node<T>(height, width, {}, name, 0)
    {
    }
    Input(std::pair<uint32_t, uint32_t> shape, const std::string& name = "Input")
        : Node<T>(shape, {}, name, 0)
    {
    }
    void compute() override {}

    void backward(const Matrix<T>*) override {}
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
    os << p.name << " Weights: " << *(Matrix<Ta>*)(&p) << "\n"
       << p.name << " Grads: " << p.grads << std::endl;
    return os;
}

#endif  // NODE_PARAMETER_HPP
