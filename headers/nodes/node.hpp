#ifndef NODE_HPP
#define NODE_HPP

#include "functors.cuh"
#include "logger.hpp"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "matrix_ops_cpu.hpp"
#include "parameter.hpp"
#include "types"

template <typename T>
struct Node;

template <typename T = FloatT>
using NodePtr = Node<T>*;

template <typename T = FloatT>
using NodePtrs = std::vector<NodePtr<T>>;

template <typename T = FloatT>
using NodePtrList = std::initializer_list<const NodePtr<T>>;

struct NodeBase
{
    NodeBase(const std::string& name, const Shape& shape)
    {
        (void)(shape);  // no warn if LOG_TRACE is disabled
        (void)(name);   // no warn if LOG_TRACE is disabled
        LOG_TRACE("Creating Node ", name, " with shape: ", shape);
        all_nodes.push_back(this);
    }

    virtual ~NodeBase() = default;

    static std::vector<NodeBase*> all_nodes;
};

template <typename T = FloatT>
struct Node : public Matrix<T>, NodeBase
{
    // TODO: Names are inconsistent, fix them
    Node(Shape s, const NodePtrs<T>& prevs, const std::string& name_, uint32 prev_count)
        : Matrix<T>(s, name_), NodeBase(this->name, s)
    {
        if (prevs.size() != prev_count)
            throw_rte_with_backtrace("Expected ", prev_count, " input(s), for ", name_, " got ",
                                     prevs.size());
        for (auto& p : prevs)
        {
            if (p == nullptr) throw_rte_with_backtrace("Input node is nullptr");
            prev_nodes.push_back(p);
        }
    }

    // call compute on all previous nodes to populate their outputs, then call forward
    virtual void compute(uint32 depth = 0)
    {
        LOG_TRACE("Computing inputs for `", this->name, "` : ", depth);
        for (auto& p : prev_nodes) p->compute(depth + 1);
        this->forward();
    }

    virtual void forward() = 0;  // Assumes that all `prev_nodes` are completed forward pass.
    virtual void backward(const Matrix<T>* e) = 0;
    virtual void update_weights(FloatT lr)
    {
        for (auto& n : prev_nodes) n->update_weights(lr);
        for (auto& p : params) p->update(lr);
        if (this->get_terminal_node()) this->get_terminal_node()->update_weights(lr);
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

    bool is_training = true;

    void set_is_training(bool is_training)
    {
        this->is_training = is_training;
        for (auto& p : params) p->set_is_training(is_training);
        for (auto& p : prev_nodes) p->set_is_training(is_training);
    }

    virtual NodePtr<T> get_terminal_node() { return nullptr; }
    virtual std::string dot_repr() { return " [label=\"" + this->name + "\" shape=rect]\n"; }

    virtual void save_weights(std::ostream&) const {}
    virtual void load_weights(std::istream&) {}
};

template <typename T = FloatT>
void graph_to_dot(NodePtr<T> node, std::string filename)
{
    NodePtrs<T> nodes;
    nodes.push_back(node);
    std::vector<std::string> edge_node_strs;

    auto make_edge = [](NodePtr<T> a, NodePtr<T> b, float32 weight = 3.f) {
        char edge_buffer[256];
        snprintf(edge_buffer, 256, "%d -> %d [label=\"%dx%dx%d\" weight=%2.1f]", a->id, b->id,
                 a->shape[2], a->shape[1], a->shape[0], weight);
        return std::string(edge_buffer);
    };

    while (!nodes.empty())
    {
        auto* n = nodes.back();
        nodes.pop_back();
        for (auto* p : n->prev_nodes)
        {
            edge_node_strs.push_back(make_edge(p, n));
            nodes.push_back(p);
        }
        edge_node_strs.push_back(std::to_string(n->id) + n->dot_repr());
        auto* terminal = n->get_terminal_node();
        if (terminal)
        {
            nodes.push_back(terminal);
            auto term_edge = make_edge(n, terminal, 1.f);
            term_edge = "\nedge [style=dotted arrowhead=none]\n" + term_edge +
                        "\nedge [style=normal arrowhead=normal];\n";
            edge_node_strs.push_back(term_edge);
        }
    }
    std::ofstream os(filename);
    os << "digraph G {\n compound=true;\n";
    std::copy(edge_node_strs.begin(), edge_node_strs.end(),
              std::ostream_iterator<std::string>(os, "\n"));
    os << '}' << std::endl;
}

template <typename Ta, typename Tb = Ta>
std::ostream& operator<<(std::ostream& os, const Parameter<Ta, Tb>& p)
{
    os << p.name << " Weights: " << *(Matrix<Ta>*)(&p) << " With Grads: " << p.grads() << std::endl;
    return os;
}

#endif  // NODE_HPP
