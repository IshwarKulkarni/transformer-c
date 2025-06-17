/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef NODE_HPP
#define NODE_HPP

#include "functors.cuh"
#include "logger.hpp"
#include "matrix.cuh"
#include "matrix_ops.hpp"
#include "matrix_ops_cpu.hpp"
#include "parameter.hpp"
#include "types"

#define LOG_NODE_NAME(...) \
    LOG(BLUE, L_JUST(this->type() + ":", 16), L_JUST(this->name, 24), __VA_ARGS__)

template <typename T>
struct Node;

template <typename T = FloatT>
using NodePtr = Node<T>*;

template <typename T>
using SharedNodePtr = std::shared_ptr<Node<T>>;

template <typename T = FloatT>
using NodePtrVec = std::vector<NodePtr<T>>;

template <typename T = FloatT>  // should be NodePtrInitList
using NodePtrList = std::initializer_list<const NodePtr<T>>;

struct NodeBase;

struct NodeBase
{
    NodeBase(const std::string& name, const Shape& shape)
    {
        (void)(shape);  // no warn if LOG_NODE_TRACE is disabled
        (void)(name);   // no warn if LOG_NODE_TRACE is disabled
        all_nodes.push_back(this);
    }

    virtual ~NodeBase() = default;

    virtual std::string type() const { return "NodeBase"; }

    static std::vector<NodeBase*> all_nodes;
};

template <typename T = FloatT>
struct Node : public Matrix<T>, NodeBase
{
    // TODO: Names are inconsistent, fix them
    Node(Shape s, const NodePtrVec<T>& prevs, const std::string& name_, uint32 prev_count)
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
    virtual void compute(Context* ctx)
    {
        LOG_NODE_TRACE(
            "", forward_count, ": Computing inputs for `", this->name, "`",
            (ctx->get_forward_pass_count() == forward_count ? ". Already computed skipping" : ""));
        if (ctx->get_forward_pass_count() == forward_count) return;  // already computed
        for (auto& p : prev_nodes)
        {
            p->compute(ctx);
        }
        forward_count++;
        this->forward(ctx);
    }

    virtual void forward(
        Context* ctx) = 0;  // Assumes that all `prev_nodes` are completed forward pass.
    virtual void backward(const Matrix<T>* e, Context* ctx) = 0;
    virtual void update_weights(FloatT lr, Context* ctx)
    {
        for (auto& p : params) p->update(lr, ctx);
    }

    std::vector<Parameter<T, T>*> params;
    NodePtrVec<T> prev_nodes{};

    uint64 forward_count = 0;
    uint64 backward_count = 0;

    Matrix<T>& prev(uint32 i)
    {
        if (i >= prev_nodes.size())
            throw_rte_with_backtrace("Index out of bounds for prev_nodes: ", i,
                                     " with size: ", prev_nodes.size(), " for node: ", this->name);
        return *((Matrix<T>*)(prev_nodes[i]));
    }

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

    // terminal node is the node that produces the output (should not be `this`)
    virtual NodePtr<T> get_terminal_node() { return nullptr; }
    virtual std::string dot_repr() { return " [label=\"" + this->name + "\" shape=rect]\n"; }

    virtual void save_weights(std::ostream&) const {}
    virtual void load_weights(std::istream&) {}

    // returns the nodes that this node depends on, separate function because
    // some nodes can have input proxies that are not "prev_nodes", but inputs
    // are read via `set_data()` like functions.
    virtual std::vector<NodePtr<T>> get_dependencies() const { return prev_nodes; }

    virtual std::string type() const { return "Node"; }
};

template <typename Ta, typename Tb = Ta>
std::ostream& operator<<(std::ostream& os, const Parameter<Ta, Tb>& p)
{
    os << *(Matrix<Ta>*)(&p) << " With Grads: " << p.grads() << std::endl;
    return os;
}

#endif  // NODE_HPP
