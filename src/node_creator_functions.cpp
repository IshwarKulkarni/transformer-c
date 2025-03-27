#include "network_builder.hpp"
#include "nodes/loss.hpp"
#include "nodes/node.hpp"
#include "nodes/parameterized.hpp"
#include "nodes/unparameterized.hpp"

std::map<std::string, NodeCreatorFunc> NodeCreatorMap::m_node_creators;

static const std::string act_key = "activation act_function common_act common_act_function";
static const std::string bias_key = "bias use_bias";
static const std::string prev_key = "prev input input_node";
static const std::string batch_key = "batch batch_size";
static const std::string dim_key = "out_size output_size out_dim output_dim";

// define NodeCreatorFunc for each node type
NodePtr<FloatT> create_input_node(std::istream& is, const std::string& name,
                                  NetworkBuilder& builder)
{
    string_pair_vec params = {
        {batch_key, ""}, {"height num_samples", ""}, {"width row_vec_size", ""}};
    builder.read_params(is, name, params);
    return new Input<FloatT>(builder.get_value<uint32>(params[0].second),
                             builder.get_value<uint32>(params[1].second),
                             builder.get_value<uint32>(params[2].second), name);
}

NodePtr<FloatT> create_dropout_node(std::istream& is, const std::string& name,
                                    NetworkBuilder& builder)
{
    string_pair_vec params = {{"p rate dropout_rate", ".25"}, {prev_key, ""}};
    builder.read_params(is, name, params);
    return new Dropout<FloatT>(builder.get_value<float>(params[0].second),
                               builder.get_node(params[1].second), name);
}

NodePtr<FloatT> create_linear_node(std::istream& is, const std::string& name,
                                   NetworkBuilder& builder)
{
    string_pair_vec params = {{dim_key, ""}, {prev_key, ""}, {bias_key, "true"}, {act_key, "relu"}};

    builder.read_params(is, name, params);

    const auto& odim = builder.get_value<uint32>(params[0].second);
    const auto& prev = builder.get_node(params[1].second);
    const auto& bias = builder.get_value<bool>(params[2].second);
    const auto& act = params[3].second;

    if (act == "relu")
        return new Linear<FloatT, Relu<FloatT>>(odim, prev, bias, name);
    else if (act == "sigmoid")
        return new Linear<FloatT, Sigmoid<FloatT>>(odim, prev, bias, name);
    else if (act == "tanh")
        return new Linear<FloatT, TanH<FloatT>>(odim, prev, bias, name);
    else if (act == "identity")
        return new Linear<FloatT, IActivation<FloatT>>(odim, prev, bias, name);
    else if (act == "leaky_relu" || act == "leakyrelu")
        return new Linear<FloatT, LeakyRelu<FloatT>>(odim, prev, bias, name);

    throw_rte_with_backtrace("Unknown activation function: ", act);
    return nullptr;
}

NodePtr<FloatT> create_self_attention_node(std::istream& is, const std::string& name,
                                           NetworkBuilder& builder)
{
    string_pair_vec params = {{dim_key, ""}, {prev_key, ""}, {bias_key, "true"}, {act_key, "relu"}};
    builder.read_params(is, name, params);
    const auto& odim = builder.get_value<uint32>(params[0].second);
    const auto& prev = builder.get_node(params[1].second);
    const auto& bias = builder.get_value<bool>(params[2].second);
    const auto& act = params[3].second;

    if (act == "relu")
        return new SelfAttention<FloatT, Relu<FloatT>>(odim, prev, bias, name);
    else if (act == "sigmoid")
        return new SelfAttention<FloatT, Sigmoid<FloatT>>(odim, prev, bias, name);
    else if (act == "tanh")
        return new SelfAttention<FloatT, TanH<FloatT>>(odim, prev, bias, name);
    else if (act == "identity")
        return new SelfAttention<FloatT, IActivation<FloatT>>(odim, prev, bias, name);
    else if (act == "leaky_relu" || act == "leakyrelu")
        return new SelfAttention<FloatT, LeakyRelu<FloatT>>(odim, prev, bias, name);

    throw_rte_with_backtrace("Unknown activation function: ", act);
    return nullptr;
}

NodePtr<FloatT> create_l2_loss_node(std::istream& is, const std::string& name,
                                    NetworkBuilder& builder)
{
    string_pair_vec params = {{"predictions", ""}, {"target", ""}};
    builder.read_params(is, name, params);
    return new L2Loss<FloatT>(
        {builder.get_node(params[0].second), builder.get_node(params[1].second)}, name);
}

NodePtr<FloatT> create_attention_node(std::istream& is, const std::string& name,
                                      NetworkBuilder& builder)
{
    string_pair_vec params = {{"q_size q_output_size q_out_dim", ""},
                              {"q_input q_prev q_input_node", ""},
                              {"q_bias", "true"},

                              {"k_size k_output_size k_out_dim", ""},
                              {"k_input k_prev k_input_node", ""},
                              {"k_bias", "true"},

                              {"v_size v_output_size v_out_dim", ""},
                              {"v_input v_prev v_input_node", ""},
                              {"v_bias", "true"},

                              {act_key + " act_function common_act common_act_function", "relu"}};
    builder.read_params(is, name, params);

    uint32 sizes[3];
    NodePtr<FloatT> prevs[3];
    bool biases[3];

    for (uint32 i = 0; i < 3; ++i)
    {
        sizes[i] = builder.get_value<uint32>(params[i * 3].second);
        prevs[i] = builder.get_node(params[i * 3 + 1].second);
        biases[i] = builder.get_value<bool>(params[i * 3 + 2].second);
    }

    LinearInputT<FloatT> q_inp = {sizes[0], prevs[0], biases[0], name + "_q"};
    LinearInputT<FloatT> k_inp = {sizes[1], prevs[1], biases[1], name + "_k"};
    LinearInputT<FloatT> v_inp = {sizes[2], prevs[2], biases[2], name + "_v"};

    const auto& act_str = params[9].second;

    if (act_str == "relu")
        return new Attention<FloatT, Relu<FloatT>>(q_inp, k_inp, v_inp, name);
    else if (act_str == "sigmoid")
        return new Attention<FloatT, Sigmoid<FloatT>>(q_inp, k_inp, v_inp, name);
    else if (act_str == "tanh")
        return new Attention<FloatT, TanH<FloatT>>(q_inp, k_inp, v_inp, name);
    else if (act_str == "identity")
        return new Attention<FloatT, IActivation<FloatT>>(q_inp, k_inp, v_inp, name);
    else if (act_str == "leaky_relu" || act_str == "leakyrelu")
        return new Attention<FloatT, LeakyRelu<FloatT>>(q_inp, k_inp, v_inp, name);
    else
        throw_rte_with_backtrace("Unknown activation function: ", act_str);
}

// Initialize all node creator functions
void initialize_node_creators()
{
    NodeCreatorMap::register_func("Input", create_input_node);
    NodeCreatorMap::register_func("Dropout", create_dropout_node);
    NodeCreatorMap::register_func("Linear", create_linear_node);
    NodeCreatorMap::register_func("SelfAttention", create_self_attention_node);
    NodeCreatorMap::register_func("L2Loss", create_l2_loss_node);
    NodeCreatorMap::register_func("Attention", create_attention_node);
}
