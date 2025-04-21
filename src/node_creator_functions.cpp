/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <stdexcept>
#include "network_graph.hpp"
#include "nodes/loss.hpp"
#include "nodes/node.hpp"
#include "nodes/parameterized.hpp"
#include "nodes/parameterized_composite.hpp"
#include "nodes/unparameterized.hpp"

std::map<std::string, NodeCreatorFunc> NodeCreatorMap::m_node_creators;

static const StringStringMap linear_params = {
    {"out_dim", ""}, {"prev", ""}, {"bias", "true"}, {"act", "relu"}};
static const StringStringMap dropout_params = {{"rate", ".25"}, {"prev", ""}};

static StringStringMap add_key_prefix(const StringStringMap& params, const std::string& prefix)
{
    StringStringMap new_params;
    for (auto& [key, value] : params)
    {
        new_params[prefix + key] = value;
    }
    return new_params;
}

// get the value of a parameter from the map, if not found or empty, return default_value
static std::string get_value(const StringStringMap& params, const std::string& key,
                             const std::string& default_value)
{
    auto it = params.find(key);
    if (it != params.end() && !it->second.empty()) return it->second;
    return default_value;
}

template <typename T>
LinearInput<T> get_linear_input(NetworkGraph& graph, const StringStringMap& params,
                                const std::string& name, std::string default_dim = "",
                                std::string default_bias = "false",
                                std::string default_act = "identity")
{
    auto prev_it = params.find("prev");
    if (prev_it == params.end()) throw_rte_with_backtrace("Linear: prev is not specified");

    const auto& prev = graph.get_node(prev_it->second);

    if (default_dim == "") default_dim = std::to_string(prev->width());
    // match these to the linear_params
    const auto& odim = graph.get_value<uint32>(get_value(params, "out_dim", default_dim));
    const auto& bias = graph.get_value<bool>(get_value(params, "bias", default_bias));
    const auto& act = get_value(params, "act", default_act);
    return LinearInput<T>{odim, prev, bias, act, name};
}

// define NodeCreatorFunc for each node type
NodePtr<FloatT> create_input_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    StringStringMap params = {{"batch", ""}, {"height", ""}, {"width", ""}};
    graph.read_params(is, name, params);
    return new Input<FloatT>(graph.get_value<uint32>(params["batch"]),
                             graph.get_value<uint32>(params["height"]),
                             graph.get_value<uint32>(params["width"]), name);
}

NodePtr<FloatT> create_dropout_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    StringStringMap params = dropout_params;
    graph.read_params(is, name, params);
    return new Dropout<FloatT>(graph.get_value<float32>(get_value(params, "rate", ".25")),
                               graph.get_node(params["prev"]), name);
}

NodePtr<FloatT> create_l2_loss_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    StringStringMap params = {{"predictions", ""}, {"target", ""}};
    graph.read_params(is, name, params);
    return new L2Loss<FloatT>(
        {graph.get_node(params["predictions"]), graph.get_node(params["target"])}, name);
}

NodePtr<FloatT> create_mean_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    StringStringMap params = {{"prev", ""}, {"dim", "0"}};
    graph.read_params(is, name, params);
    NodePtr<FloatT> prev = graph.get_node(params["prev"]);
    uint32 dim = graph.get_value<uint32>(params["dim"]);
    if (dim == 0)
        return new Mean<FloatT, 0>(prev, name);
    else if (dim == 1)
        return new Mean<FloatT, 1>(prev, name);
    else if (dim == 2)
        return new Mean<FloatT, 2>(prev, name);
    else
        throw_rte_with_backtrace("Mean: dim must be 0, 1, or 2");
}

NodePtr<FloatT> create_linear_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    StringStringMap params = linear_params;

    graph.read_params(is, name, params);
    return new Linear<FloatT>(get_linear_input<FloatT>(graph, params, name));
}

NodePtr<FloatT> create_attention_node(std::istream& is, const std::string& name,
                                      NetworkGraph& graph)
{
    StringStringMap q_params = add_key_prefix(linear_params, "q_");
    StringStringMap k_params = add_key_prefix(linear_params, "k_");
    StringStringMap v_params = add_key_prefix(linear_params, "v_");
    graph.read_params(is, name, q_params);
    graph.read_params(is, name, k_params);
    graph.read_params(is, name, v_params);

    return new Attention<FloatT>(
        get_linear_input<FloatT>(graph, q_params, name + "_q"),
        get_linear_input<FloatT>(graph, k_params, name + "_k", q_params["dim"]),
        get_linear_input<FloatT>(graph, v_params, name + "_v", q_params["dim"]));
}

NodePtr<FloatT> create_self_attention_node(std::istream& is, const std::string& name,
                                           NetworkGraph& graph)
{
    StringStringMap params = linear_params;
    graph.read_params(is, name, params);
    return new SelfAttention<FloatT>(get_linear_input<FloatT>(graph, params, name));
}

NodePtr<FloatT> create_cross_attention_node(std::istream& is, const std::string& name,
                                            NetworkGraph& graph)
{
    StringStringMap q_params = add_key_prefix(linear_params, "q_");
    StringStringMap kv_params = add_key_prefix(linear_params, "kv_");
    graph.read_params(is, name, q_params);
    graph.read_params(is, name, kv_params);

    LinearInput<FloatT> q_inp = get_linear_input<FloatT>(graph, q_params, name + "_q");
    LinearInput<FloatT> kv_inp = get_linear_input<FloatT>(graph, kv_params, name + "_kv");

    return new CrossAttention<FloatT>(q_inp, kv_inp, name);
}

NodePtr<FloatT> create_multi_head_attention_node(std::istream& is, const std::string& name,
                                                 NetworkGraph& graph)
{
    StringStringMap q_params = add_key_prefix(linear_params, "q_");
    StringStringMap k_params = add_key_prefix(linear_params, "k_");
    StringStringMap v_params = add_key_prefix(linear_params, "v_");
    StringStringMap output_params = add_key_prefix(linear_params, "output_");
    StringStringMap params = {{"num_heads", "2"}};

    graph.read_params(is, name, q_params);
    graph.read_params(is, name, k_params);
    graph.read_params(is, name, v_params);
    graph.read_params(is, name, output_params);
    graph.read_params(is, name, params);

    const auto& num_heads = graph.get_value<uint32>(params["num_heads"]);

    LinearInput<FloatT> q_inp = get_linear_input<FloatT>(graph, q_params, name + "_q");
    LinearInput<FloatT> k_inp =
        get_linear_input<FloatT>(graph, k_params, name + "_k", q_params["dim"]);
    LinearInput<FloatT> v_inp =
        get_linear_input<FloatT>(graph, v_params, name + "_v", q_params["dim"]);
    LinearInput<FloatT> output_inp =
        get_linear_input<FloatT>(graph, output_params, name + "_output");

    return new MultiHeadAttention<FloatT>(num_heads, q_inp, k_inp, v_inp, output_inp, name);
}

NodePtr<FloatT> create_multi_head_self_attention_node(std::istream& is, const std::string& name,
                                                      NetworkGraph& graph)
{
    StringStringMap qkv_params = linear_params;
    StringStringMap out_params = add_key_prefix(linear_params, "output_");
    StringStringMap params = {{"num_heads", "2"}};

    graph.read_params(is, name, qkv_params);
    graph.read_params(is, name, out_params);
    graph.read_params(is, name, params);

    return new MultiHeadSelfAttention<FloatT>(
        graph.get_value<uint32>(params["num_heads"]),
        get_linear_input<FloatT>(graph, qkv_params, name),
        get_linear_input<FloatT>(graph, out_params, name, qkv_params["dim"]), name);
}

NodePtr<FloatT> create_multi_head_cross_attention_node(std::istream& is, const std::string& name,
                                                       NetworkGraph& graph)
{
    StringStringMap q_params = add_key_prefix(linear_params, "q_");
    StringStringMap kv_params = add_key_prefix(linear_params, "kv_");
    StringStringMap out_params = add_key_prefix(linear_params, "output_");
    StringStringMap params = {{"num_heads", "2"}};

    graph.read_params(is, name, q_params);
    graph.read_params(is, name, kv_params);
    graph.read_params(is, name, out_params);
    graph.read_params(is, name, params);

    return new MultiHeadCrossAttention<FloatT>(
        graph.get_value<uint32>(params["num_heads"]),
        get_linear_input<FloatT>(graph, q_params, name),
        get_linear_input<FloatT>(graph, kv_params, name, q_params["dim"]),
        get_linear_input<FloatT>(graph, out_params, name, q_params["dim"]), name);
}

NodePtr<FloatT> create_sine_positional_embedding_node(std::istream& is, const std::string& name,
                                                      NetworkGraph& graph)
{
    StringStringMap params = {{"prev", ""}};
    graph.read_params(is, name, params);
    return new SinePositionalEmbedding<FloatT>(graph.get_node(params["prev"]), name);
}

NodePtr<FloatT> create_feed_forward_node(std::istream& is, const std::string& name,
                                         NetworkGraph& graph)
{
    StringStringMap l1_params = linear_params;
    StringStringMap params = {{"rate1", ".25"}, {"rate2", ".25"}, {"intermediate_dim", ""}};

    graph.read_params(is, name, l1_params);
    graph.read_params(is, name, params);

    LinearInput<FloatT> l1_inp = get_linear_input<FloatT>(graph, l1_params, name + "_l1");

    if (params["intermediate_dim"] != "")
    {
        if (l1_params["dim"] != params["intermediate_dim"])
            throw_rte_with_backtrace("FeedForward: intermediate_dim and l1_dim must be the same");

        return new FeedForward<FloatT>(l1_inp, graph.get_value<uint32>(params["intermediate_dim"]),
                                       graph.get_value<float32>(params["rate1"]),
                                       graph.get_value<float32>(params["rate2"]), name);
    }
    else
    {
        return new FeedForward<FloatT>(l1_inp, graph.get_value<float32>(params["rate1"]),
                                       graph.get_value<float32>(params["rate2"]), name);
    }
}

// Initialize all node creator functions
void initialize_node_creators()
{
    NodeCreatorMap::register_func("Input", create_input_node);
    NodeCreatorMap::register_func("Dropout", create_dropout_node);
    NodeCreatorMap::register_func("L2Loss", create_l2_loss_node);
    NodeCreatorMap::register_func("Linear", create_linear_node);
    NodeCreatorMap::register_func("Attention", create_attention_node);
    NodeCreatorMap::register_func("SelfAttention", create_self_attention_node);
    NodeCreatorMap::register_func("CrossAttention", create_cross_attention_node);
    NodeCreatorMap::register_func("MultiHeadAttention", create_multi_head_attention_node);
    NodeCreatorMap::register_func("MultiHeadSelfAttention", create_multi_head_self_attention_node);
    NodeCreatorMap::register_func("MultiHeadCrossAttention",
                                  create_multi_head_cross_attention_node);
    NodeCreatorMap::register_func("SinePositionalEmbedding", create_sine_positional_embedding_node);
    NodeCreatorMap::register_func("FeedForward", create_feed_forward_node);
}
