#ifndef NODE_PARAMETER_HPP
#define NODE_PARAMETER_HPP

#include "matrix.cuh"
#include "matrix_ops.cuh"

template <typename TW, typename TG = TW>  // weight and gradient
struct Parameter : Matrix<TW>
{
    Matrix<TG> grads;
    void update(float32 lr)
    {
        WeightUpdate<TW, TG> updateFunc(lr);
        binary_apply(updatedWeights, *this, grads, updateFunc);
        std::swap(this->data, updatedWeights.data);
        fill(grads, (TG*)nullptr);
    }
    Parameter(uint32_t height, uint32_t width, std::string name = "Param")
        : Matrix<TW>(xavier_init<TW>(height, width, name)),
          grads(height, width, name + "_grads"),
          updatedWeights(height, width, name + "_updated")
    {
        fill(updatedWeights, (TW*)nullptr);
        fill(grads, (TW*)nullptr);
    }

 private:
    Matrix<TW> updatedWeights;
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
    Node(uint32_t height, uint32_t width, const NodePtrs<T>& prev, const std::string& name_,
         uint32 prev_count)
        : Matrix<T>(height, width, name_)
    {
        if (prev.size() != prev_count)
            throw_rte_with_backtrace("Expected ", prev_count, " input(s), for ", name_, " got ",
                                     prev_nodes.size());
        for (auto& p : prev)
        {
            if (p == nullptr) throw_rte_with_backtrace("Input node is nullptr");
            prev_nodes.push_back(p);
        }
    }

    Node(std::pair<uint32_t, uint32_t> shape, const NodePtrs<T>& prev, const std::string& name_,
         uint32 prev_count)
        : Node(shape.first, shape.second, prev, name_, prev_count)
    {
    }

    virtual void
    compute()  // call compute on all previous nodes to populate their outputs, then call forward
    {
        for (auto& p : prev_nodes)
        {
            p->compute();
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

    virtual uint32 n_trainable_params(std::set<Parameter<T, T>*>& seen = {})
    {
        uint32 total = 0;

        for (auto& p : params)
        {
            if (seen.find(p) != seen.end()) continue;
            seen.insert(p);
            total += p->numels();
        }
        for (auto& n : prev_nodes)
        {
            total += n->n_trainable_params(seen);
        }
        return total;
    }
    virtual std::string dot_repr() { return " [label=\"" + this->name + "\"]"; }
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
            snprintf(edge_buffer, 256, "%d -> %d [label=\"%s\", splines=\"ortho\"]\n", p->id, n->id,
                     p->shape_str.c_str());
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
    os << p.name << " Weights: " << *(Matrix<Ta>*)(&p) << "\n"
       << p.name << " Grads: " << p.grads << std::endl;
    return os;
}

#endif  // NODE_PARAMETER_HPP
