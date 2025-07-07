/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <iostream>
#include "network_graph.hpp"
#include "nodes/loss.hpp"
#include "nodes/parameterized.hpp"
#include "nodes/parameterized_composite.hpp"
#include "nodes/unparameterized.hpp"
#include "string_utils.hpp"

// reads from `is` and populates m_param_defaults, overwriting existing values,
// fails if any param read from `is` is not over writting m_param_defaults, or if
// and value in m_param_defaults is still NoneString
void NodeCreator::populate_params(NetworkGraph& graph, std::istream& is,
                                  const std::string& node_name)
{
    const StringStringMap& params = graph.get_key_value_pairs(is, node_name);
    for (const auto& [key, value] : params)
    {
        auto it = m_param_defaults.find(key);
        if (it == m_param_defaults.end())
        {
            const auto& param_names = get_param_names();
            auto [closest_match, dist] =
                get_closest_match(param_names.begin(), param_names.end(), key);
            std::string perhaps = dist < 0.5 ? YELLOW "\n\t`" + closest_match + "` perhaps?" : "";
            throw_rte_with_backtrace("Parameter `", key, "` unacceptable for node '", NodeType,
                                     "'. Has to be one of:\n", join(param_names, "\n"), perhaps);
        }
        m_param_defaults[key] = value;
    }
    for (const auto& [key, value] : m_param_defaults)
    {
        if (value == None)
        {
            throw_rte_with_backtrace("Parameter ", key, " is not provided for node ", node_name);
        }
    }
}

NodePtr<FloatT> NodeCreator::get_prev_node(NetworkGraph& graph)
{
    if (m_param_defaults.find("prev") == m_param_defaults.end())
    {
        throw_rte_with_backtrace("Node ", NodeType, " does not take a previous node");
    }
    std::string prev_name = m_param_defaults["prev"];
    if (auto prev = graph.get_node(prev_name)) return prev;
    throw_rte_with_backtrace("Previous node ", prev_name, " not found for node ", NodeType);
}

template <typename T>
T NodeCreator::get_value(NetworkGraph& graph, const std::string& variable_name)
{
    auto val = this->get_param_value(variable_name);
    if (auto opt = graph.get_value_optionally<T>(val)) return *opt;
    throw_rte_with_backtrace("Parameter ", variable_name, "cannot be resolved.\n");
}

// bool specialization of get_value
template <>
bool NodeCreator::get_value<bool>(NetworkGraph& graph, const std::string& variable_name)
{
    auto val = this->get_param_value(variable_name);
    if (auto opt = graph.get_value_optionally<bool>(val)) return *opt;

    // parse defualt value in m_param_defaults
    if (val == "true" || val == "1" || val == "yes" || val == "on") return true;
    if (val == "false" || val == "0" || val == "no" || val == "off") return false;

    throw_rte_with_backtrace("Parameter ", variable_name, " has invalid value: ", val, " for node ",
                             NodeType);
}

// string specialization of get_value
template <>
std::string NodeCreator::get_value<std::string>(NetworkGraph& graph,
                                                const std::string& variable_name)
{
    auto val = this->get_param_value(variable_name);
    if (auto opt = graph.get_value_optionally<std::string>(val)) return *opt;
    throw_rte_with_backtrace("Parameter ", variable_name, "cannot be resolved.\n");
}

class InputNodeCreator : public NodeCreator
{
 public:
    InputNodeCreator()
        : NodeCreator("Input", {{"batch", "0"}, {"seq_len", "1"}, {"row_size", "0"}}, false)
    {
    }

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        return new Input<FloatT>(get_value<uint32>(graph, "batch"),
                                 get_value<uint32>(graph, "seq_len"),
                                 get_value<uint32>(graph, "row_size"), name);
    }
};

// Dropout Node Creator
class DropoutNodeCreator : public NodeCreator
{
 public:
    DropoutNodeCreator() : NodeCreator("Dropout", {{"rate", "0.2"}}) {}

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        return new Dropout<FloatT>(get_value<float32>(graph, "rate"), get_prev_node(graph), name);
    }
};

// Softmax Node Creator
class SoftmaxNodeCreator : public NodeCreator
{
 public:
    SoftmaxNodeCreator() : NodeCreator("Softmax", {{"dim", "0"}}) {}

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        auto prev = get_prev_node(graph);
        auto dim = get_value<uint32>(graph, "dim");

        if (dim == 0)
            return new SoftmaxDim0<FloatT>(prev, name);
        else if (dim == 1)
            return new SoftmaxDim1<FloatT>(prev, name);
        throw_rte_with_backtrace("Invalid dimension for softmax node: ", dim);
    }
};

// Template for Loss2 Node Creator
template <typename LossClass>
class Loss2NodeCreator : public NodeCreator
{
 public:
    Loss2NodeCreator(std::string node_type)
        : NodeCreator(node_type, {{"predictions", "[None]"}, {"target", "[None]"}}, false)
    {
    }

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        auto pred_name = get_value<std::string>(graph, "predictions");
        auto target_name = get_value<std::string>(graph, "target");

        return new LossClass({graph.get_node(pred_name), graph.get_node(target_name)}, name);
    }
};

// L2Loss Node Creator
class L2LossNodeCreator : public Loss2NodeCreator<L2Loss<FloatT>>
{
 public:
    L2LossNodeCreator() : Loss2NodeCreator<L2Loss<FloatT>>("L2Loss") {}
};

// LogSoftmaxCELoss Node Creator
class LogSoftmaxCELossNodeCreator : public Loss2NodeCreator<LogSoftmaxCELoss<FloatT>>
{
 public:
    LogSoftmaxCELossNodeCreator() : Loss2NodeCreator<LogSoftmaxCELoss<FloatT>>("LogSoftmaxCELoss")
    {
    }
};

// NLLLoss Node Creator
class NLLLossNodeCreator : public Loss2NodeCreator<NLLLoss<FloatT>>
{
 public:
    NLLLossNodeCreator() : Loss2NodeCreator<NLLLoss<FloatT>>("NLLLoss") {}
};

// Linear Node Creator
class LinearNodeCreator : public NodeCreator
{
 public:
    LinearNodeCreator()
        : NodeCreator("Linear", {{"out_dim", "0"}, {"bias", "1"}, {"act", "identity"}})
    {
    }

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        LinearInput<FloatT> input = {.out_size = get_value<uint32>(graph, "out_dim"),
                                     .prev = get_prev_node(graph),
                                     .useBias = get_value<bool>(graph, "bias"),
                                     .act_name = get_value<std::string>(graph, "act"),
                                     .name = name};
        return new Linear<FloatT>(input);
    }
};

// SelfAttention Node Creator
class SelfAttentionNodeCreator : public NodeCreator
{
 public:
    SelfAttentionNodeCreator()
        : NodeCreator("SelfAttention", {{"out_dim", None}, {"bias", "1"}, {"act", "identity"}})
    {
    }

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        const LinearInput<FloatT> qkv = {.out_size = get_value<uint32>(graph, "out_dim"),
                                         .prev = get_prev_node(graph),
                                         .useBias = get_value<bool>(graph, "bias"),
                                         .act_name = get_value<std::string>(graph, "act"),
                                         .name = name};

        return new SelfAttention<FloatT>(qkv, name);
    }
};

// MultiHeadSelfAttention Node Creator
class MultiHeadSelfAttentionNodeCreator : public NodeCreator
{
 public:
    MultiHeadSelfAttentionNodeCreator()
        : NodeCreator("MultiHeadSelfAttention", {{"out_dim", None},
                                                 {"bias", "1"},
                                                 {"act", "identity"},
                                                 {"qkv_dims", None},
                                                 {"qkv_bias", "0"},
                                                 {"qkv_act", "identity"},
                                                 {"num_heads", "1"}})
    {
    }

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        LinearInput<FloatT> output = {.out_size = get_value<uint32>(graph, "out_dim"),
                                      .useBias = get_value<bool>(graph, "bias"),
                                      .act_name = get_value<std::string>(graph, "act"),
                                      .name = name + "_out"};

        auto qkv_dim = std::to_string(output.out_size);

        LinearInput<FloatT> qkv = {.out_size = get_value<uint32>(graph, "qkv_dims"),
                                   .prev = get_prev_node(graph),
                                   .useBias = get_value<bool>(graph, "qkv_bias"),
                                   .act_name = get_value<std::string>(graph, "qkv_act"),
                                   .name = name};

        uint32 num_heads = get_value<uint32>(graph, "num_heads");

        return new MultiHeadSelfAttention<FloatT>(num_heads, qkv, output, name);
    }
};

// FeedForward Node Creator
class FeedForwardNodeCreator : public NodeCreator
{
 public:
    FeedForwardNodeCreator()
        : NodeCreator("FeedForward", {{"intermediate_dim", None},
                                      {"out_dim", None},
                                      {"l1_bias", "1"},
                                      {"l1_act", "identity"},
                                      {"l2_bias", "1"},
                                      {"l2_act", "identity"},
                                      {"rate1", "0.2"},
                                      {"rate2", "0.2"},
                                      {"normalize", "false"}})
    {
    }

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        LinearInput<FloatT> l1_input = {.prev = get_prev_node(graph),
                                        .useBias = get_value<bool>(graph, "l1_bias"),
                                        .act_name = get_value<std::string>(graph, "l1_act"),
                                        .name = name + "_l1"};

        float32 p1 = get_value<float>(graph, "rate1");
        uint32 intermediate_dim = get_value<uint32>(graph, "intermediate_dim");

        std::string normalize_str = get_value<std::string>(graph, "normalize");
        uint32 normalize_dim = UINT32_MAX;
        if (normalize_str == "width")
            normalize_dim = 0;
        else if (normalize_str == "height")
            normalize_dim = 1;
        else if (normalize_str == "batch")
            normalize_dim = 2;
        else if (normalize_str == "false" || normalize_str == "no" || normalize_str == "0" ||
                 normalize_str == "none")
            normalize_dim = UINT32_MAX;
        else
            throw_rte_with_backtrace("Invalid normalize value: ", normalize_str,
                                     "should be one of: width, height, batch, false/0/none");

        LinearInput<FloatT> l2_input = {.out_size = get_value<uint32>(graph, "out_dim"),
                                        .useBias = get_value<bool>(graph, "l2_bias"),
                                        .act_name = get_value<std::string>(graph, "l2_act"),
                                        .name = name + "_l2"};

        float32 p2 = get_value<float>(graph, "rate2");

        return new FeedForward<FloatT>(l1_input, p1, intermediate_dim, l2_input, p2, normalize_dim,
                                       name);
    }
};

// SinePositionalEmbedding Node Creator
class SinePositionalEmbeddingNodeCreator : public NodeCreator
{
 public:
    SinePositionalEmbeddingNodeCreator() : NodeCreator("SinePositionalEmbedding", {}) {}

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        return new SinePositionalEmbedding<FloatT>(get_prev_node(graph));
    }
};

// Mean Node Creator
class MeanNodeCreator : public NodeCreator
{
 public:
    MeanNodeCreator() : NodeCreator("Mean", {{"dim", "0"}}) {}

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        auto prev = get_prev_node(graph);
        auto dim = get_value<uint32>(graph, "dim");

        if (dim == 0)
            return new Mean<FloatT, 0>(prev, name);
        else if (dim == 1)
            return new Mean<FloatT, 1>(prev, name);
        else if (dim == 2)
            return new Mean<FloatT, 2>(prev, name);
        else
            throw_rte_with_backtrace("Invalid dimension for mean node: ", dim);
    }
};

// Norm Node Creator
class NormNodeCreator : public NodeCreator
{
 public:
    NormNodeCreator() : NodeCreator("Norm", {{"dim", "0"}}) {}

    NodePtr<FloatT> create(std::istream& is, const std::string& name, NetworkGraph& graph) override
    {
        this->populate_params(graph, is, name);
        auto prev = get_prev_node(graph);
        auto dim = get_value<uint32>(graph, "dim");

        if (dim == 0)
            return new Normalize<FloatT, 0>(prev, name);
        else if (dim == 1)
            return new Normalize<FloatT, 1>(prev, name);
        else if (dim == 2)
            return new Normalize<FloatT, 2>(prev, name);
        else
            throw_rte_with_backtrace("Invalid dimension for mean node: ", dim);
    }
};

NodeCreator* NodeCreatorMap::get(const std::string& name)
{
    if (m_node_creators.find(name) == m_node_creators.end())
    {
        const auto& creator_names = get_creator_names();
        auto [closest_match, dist] =
            get_closest_match(creator_names.begin(), creator_names.end(), name);
        throw_rte_with_backtrace("Node creator function for `", name,
                                 "` is not defined. Did you mean `", closest_match, "`?");
    }
    return m_node_creators[name].get();
}

std::map<std::string, std::unique_ptr<NodeCreator>> NodeCreatorMap::m_node_creators;

// Initialize all node creators
void NodeCreatorMap::initialize()
{
    if (!m_node_creators.empty()) return;
    m_node_creators["Input"] = std::make_unique<InputNodeCreator>();
    m_node_creators["Dropout"] = std::make_unique<DropoutNodeCreator>();
    m_node_creators["Softmax"] = std::make_unique<SoftmaxNodeCreator>();
    m_node_creators["L2Loss"] = std::make_unique<L2LossNodeCreator>();
    m_node_creators["LogSoftmaxCELoss"] = std::make_unique<LogSoftmaxCELossNodeCreator>();
    m_node_creators["NLLLoss"] = std::make_unique<NLLLossNodeCreator>();
    m_node_creators["LSMCE"] = std::make_unique<LogSoftmaxCELossNodeCreator>();
    m_node_creators["Linear"] = std::make_unique<LinearNodeCreator>();
    m_node_creators["Norm"] = std::make_unique<NormNodeCreator>();
    m_node_creators["SelfAttention"] = std::make_unique<SelfAttentionNodeCreator>();
    m_node_creators["MultiHeadSelfAttention"] =
        std::make_unique<MultiHeadSelfAttentionNodeCreator>();
    m_node_creators["MHSA"] = std::make_unique<MultiHeadSelfAttentionNodeCreator>();
    m_node_creators["SinePositionalEmbedding"] =
        std::make_unique<SinePositionalEmbeddingNodeCreator>();
    m_node_creators["FeedForward"] = std::make_unique<FeedForwardNodeCreator>();
    m_node_creators["Mean"] = std::make_unique<MeanNodeCreator>();
}
