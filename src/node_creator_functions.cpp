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

static const StringVector EmptyStrings = {};
std::map<std::string, NodeCreatorFunc> NodeCreatorMap::m_node_creators;

static StringVector out_dims = {"out_dim", "out_size", "dim"};
static StringVector bias_keys = {"bias"};
static StringVector act_keys = {"act", "activation"};

std::string join(const StringVector& strs, const std::string& delim = ", ")
{
    std::string result;
    for (uint32 i = 0; i < strs.size(); ++i)
    {
        result += strs[i];
        if (i < strs.size() - 1) result += delim;
    }
    return result;
}

std::string join(const StringStringMap& params, const std::string& delim = "\n")
{
    std::string result;
    for (const auto& [key, value] : params)
    {
        result += key + " : " + value + delim;
    }
    return result;
}

StringVector merge(const StringVector& a, const StringVector& b)
{
    StringVector result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

template <typename T>
Optional<T> get_value(NetworkGraph& graph, const StringStringMap& params, const std::string& key,
                      const StringVector& values)
{
    // get value for key from params, if not found, walk down the values vector
    auto it = params.find(key);

    if (it != params.end())  // defined in the block
    {
        if (auto opt = graph.get_value_optionally<T>(it->second)) return *opt;
    }
    for (const auto& value : values)
    {
        if (auto opt = graph.get_value_optionally<T>(value)) return *opt;
    }
    return Optional<T>();
}

template <typename T>  // version of above but check one of multiple key values
T get_value(NetworkGraph& graph, const StringStringMap& params, const StringVector& keys,
            const StringVector& values)
{
    for (const auto& key : keys)
    {
        if (auto opt = get_value<T>(graph, params, key, values)) return *opt;
    }

    throw_rte_with_backtrace("Value for key ", join(keys, " || "), " not found amongst\n",
                             join(params));
}

NodePtr<FloatT> get_prev_node(NetworkGraph& graph, const StringStringMap& params)
{
    auto prev_str = get_value<std::string>(graph, params, StringVector{"prev", "input"}, {});
    return graph.get_node(prev_str);
}

NodePtr<FloatT> create_input_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);
    return new Input<FloatT>(
        get_value<uint32_t>(graph, params, StringVector{"batch", "b"}, {}),
        get_value<uint32_t>(graph, params, StringVector{"num_samples", "height"}, {}),
        get_value<uint32_t>(graph, params, StringVector{"row_vec_size", "width"}, {}), name);
}

NodePtr<FloatT> create_dropout_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);
    return new Dropout<FloatT>(get_value<float>(graph, params, StringVector{"rate", "p"}, {}),
                               get_prev_node(graph, params), name);
}

NodePtr<FloatT> create_softmax_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);
    auto prev = get_prev_node(graph, params);
    auto dim = get_value<uint32_t>(graph, params,
                                   StringVector{"dim", "reduce_dim", "reduce_on_dim"}, {"0"});

    if (dim == 0)
        return new SoftmaxDim0<FloatT>(prev, name);
    else if (dim == 1)
        return new SoftmaxDim1<FloatT>(prev, name);
    throw_rte_with_backtrace("Invalid dimension for softmax node: ", dim);
}

template <typename LossClass>
NodePtr<FloatT> create_loss2_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);
    auto preds = get_value<std::string>(graph, params, StringVector{"predictions", "pred"}, {});
    auto target = get_value<std::string>(graph, params, StringVector{"target", "tgt"}, {});

    return new LossClass({graph.get_node(preds), graph.get_node(target)}, name);
}

NodePtr<FloatT> create_nll_loss_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);
    auto preds = get_value<std::string>(graph, params, StringVector{"predictions", "pred"}, {});
    auto target = get_value<std::string>(graph, params, StringVector{"target", "tgt"}, {});
    return new NLLLoss<FloatT>({graph.get_node(preds), graph.get_node(target)}, name);
}

NodePtr<FloatT> create_linear_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);

    LinearInput<FloatT> input = {
        .out_size = get_value<uint32_t>(graph, params, out_dims, {}),
        .prev = get_prev_node(graph, params),
        .useBias = get_value<bool>(graph, params, bias_keys, {"1"}),
        .act_name = get_value<std::string>(graph, params, act_keys, {"identity"}),
        .name = name};
    return new Linear<FloatT>(input);
}

NodePtr<FloatT> create_self_attention_node(std::istream& is, const std::string& name,
                                           NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);

    const LinearInput<FloatT> qkv = {
        .out_size = get_value<uint32_t>(graph, params, out_dims, {}),
        .prev = get_prev_node(graph, params),
        .useBias = get_value<bool>(graph, params, bias_keys, {"1"}),
        .act_name = get_value<std::string>(graph, params, act_keys, {"identity"}),
        .name = name};

    return new SelfAttention<FloatT>(qkv, name);
}

NodePtr<FloatT> create_multi_head_self_attention_node(std::istream& is, const std::string& name,
                                                      NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);

    LinearInput<FloatT> output = {
        .out_size = get_value<uint32_t>(graph, params, out_dims, {}),
        .prev = nullptr,  // will be ignored
        .useBias = get_value<bool>(graph, params, bias_keys, {"0"}),
        .act_name = get_value<std::string>(graph, params, act_keys, {"identity"}),
        .name = name + "_out"};

    auto qkv_dim = std::to_string(output.out_size);

    LinearInput<FloatT> qkv = {
        .out_size = get_value<uint32_t>(graph, params, merge({"qkv_dims"}, out_dims), {qkv_dim}),
        .prev = get_prev_node(graph, params),
        .useBias = get_value<bool>(graph, params, merge({"qkv_bias"}, bias_keys), {"0"}),
        .act_name =
            get_value<std::string>(graph, params, merge({"qkv_act"}, act_keys), {"identity"}),
        .name = name};

    uint32 num_heads = get_value<uint32_t>(graph, params, StringVector{"num_heads", "nheads"}, {});

    return new MultiHeadSelfAttention<FloatT>(num_heads, qkv, output, name);
}

NodePtr<FloatT> create_feed_forward_node(std::istream& is, const std::string& name,
                                         NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);

    LinearInput<FloatT> l1_input = {
        .out_size = 0,  // ignored
        .prev = get_prev_node(graph, params),
        .useBias = get_value<bool>(graph, params, merge({"l1_bias"}, bias_keys), {"1"}),
        .act_name =
            get_value<std::string>(graph, params, merge({"l1_act"}, act_keys), {"identity"}),
        .name = name + "_l1"};

    float32 p1 = get_value<float>(graph, params, StringVector{"rate1", "p1", "dropout1"}, {"0.2"});
    StringVector intermediate_dim_keys = merge({"intermediate_dim", "intermediate"}, out_dims);
    uint32 intermediate_dim = get_value<uint32_t>(graph, params, intermediate_dim_keys, {});

    LinearInput<FloatT> l2_input = {
        .out_size = get_value<uint32_t>(graph, params, merge({"l2_out_size"}, out_dims), {}),
        .prev = nullptr,  // will be ignored
        .useBias = get_value<bool>(graph, params, merge({"l2_bias"}, bias_keys), {"1"}),
        .act_name =
            get_value<std::string>(graph, params, merge({"l2_act"}, act_keys), {"identity"}),
        .name = name + "_l2"};

    float32 p2 = get_value<float>(graph, params, StringVector{"rate2", "p2", "dropout2"}, {"0.2"});

    return new FeedForward<FloatT>(l1_input, p1, intermediate_dim, l2_input, p2, name);
}

NodePtr<FloatT> create_sine_positional_embedding_node(std::istream& is, const std::string& name,
                                                      NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);
    return new SinePositionalEmbedding<FloatT>(get_prev_node(graph, params), name);
}

NodePtr<FloatT> create_mean_node(std::istream& is, const std::string& name, NetworkGraph& graph)
{
    const auto& params = graph.get_key_value_pairs(is, name);
    auto prev = get_prev_node(graph, params);
    auto dim = get_value<uint32_t>(graph, params,
                                   StringVector{"dim", "reduce_dim", "reduce_on_dim"}, {"0"});

    if (dim == 0)
        return new Mean<FloatT, 0>(prev, name);
    else if (dim == 1)
        return new Mean<FloatT, 1>(prev, name);
    else if (dim == 2)
        return new Mean<FloatT, 2>(prev, name);
    else
        throw_rte_with_backtrace("Invalid dimension for mean node: ", dim);
}

// Initialize all node creator functions
void initialize_node_creators()
{
    NodeCreatorMap::register_func("Input", create_input_node);
    NodeCreatorMap::register_func("Dropout", create_dropout_node);
    NodeCreatorMap::register_func("Softmax", create_softmax_node);
    NodeCreatorMap::register_func("L2Loss", create_loss2_node<L2Loss<FloatT>>);
    NodeCreatorMap::register_func("LogSoftmaxCELoss", create_loss2_node<LogSoftmaxCELoss<FloatT>>);
    NodeCreatorMap::register_func("NLLLoss", create_nll_loss_node);
    NodeCreatorMap::register_func("LSMCE", create_loss2_node<LogSoftmaxCELoss<FloatT>>);
    NodeCreatorMap::register_func("Linear", create_linear_node);
    // NodeCreatorMap::register_func("Attention", create_attention_node);
    NodeCreatorMap::register_func("SelfAttention", create_self_attention_node);
    // NodeCreatorMap::register_func("CrossAttention", create_cross_attention_node);
    // NodeCreatorMap::register_func("MultiHeadAttention", create_multi_head_attention_node);
    NodeCreatorMap::register_func("MultiHeadSelfAttention", create_multi_head_self_attention_node);
    NodeCreatorMap::register_func("MHSA", create_multi_head_self_attention_node);
    // NodeCreatorMap::register_func("MultiHeadCrossAttention",
    // create_multi_head_cross_attention_node);
    NodeCreatorMap::register_func("SinePositionalEmbedding", create_sine_positional_embedding_node);
    NodeCreatorMap::register_func("FeedForward", create_feed_forward_node);
    NodeCreatorMap::register_func("Mean", create_mean_node);
}
