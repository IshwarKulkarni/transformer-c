// network_builder.cpp, 2025-03-24, 10:00
// uses a simple network description language to build networks, with nodes of types of child
// classes of Node

#include "network_builder.hpp"
#include <regex>
#include "nodes/loss.hpp"
#include "nodes/node.hpp"
#include "nodes/parameterized.hpp"
#include "nodes/unparameterized.hpp"
#include "utils.hpp"

static Optional<std::pair<std::string, std::string>> parse_key_value_pair(const std::string& line)
{
    auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) return {};

    std::string key = line.substr(0, colon_pos);
    std::string value = line.substr(colon_pos + 1);
    // strip whitespace from key and value
    key = std::regex_replace(key, std::regex("^\\s+|\\s+$"), "");
    value = std::regex_replace(value, std::regex("^\\s+|\\s+$"), "");
    return std::make_pair(key, value);
}

uint32 get_line_number(std::istream& is)
{
    auto pos = is.tellg();
    is.seekg(0, std::ios::beg);
    // count the number of newlines until pos
    uint64 currentPosition = 0;
    uint64 newlineCount = 0;
    char c;
    while (is.get(c) && currentPosition++ < pos)
        if (c == '\n') newlineCount++;
    is.seekg(pos);
    return newlineCount;
}

// read subsequent lines, get key value pairs, see if key is a substring of param_names[i]
// if so, set param_values[i] to the value, if not key is not a substring of any param_names, throw
// an error if and empty line is encountered, throw an error param_names[i] is name of ith
// parameter, usually a space separated list of words that are alternate names of the parameter
string_vector NetworkBuilder::read_params(const string_vector& param_names, std::istream& is,
                                          const std::string& node_name)
{
    std::string line;
    string_vector param_values(param_names.size(), "");
    uint32 param_count = 0;
    while (getline(is, line))
    {
        line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
        if (line.empty())
        {
            if (param_count < param_names.size())
                throw_rte_with_backtrace(
                    "Premature empty line (end of block); Not all parameters were set for node ",
                    node_name, " near line ", get_line_number(is));
            break;
        }
        if (auto key_value_pair = parse_key_value_pair(line))
        {
            auto [key, value] = *key_value_pair;
            bool found = false;
            for (uint32 i = 0; i < param_names.size(); ++i)
            {
                const auto& param_name = param_names[i];
                // if key is a substring of param_name, then param_values[i] = value
                if (param_name.find(key) != std::string::npos)
                {
                    if (!param_values[i].empty())
                        throw_rte_with_backtrace("Parameter ", param_name, " for node ", node_name,
                                                 " is already set to ", param_values[i]);
                    param_values[i] = value;
                    param_count++;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                throw_rte_with_backtrace("Parameter `", key, "` for node `", node_name,
                                         "` is extraneous; near line ", get_line_number(is));
            }
            else
            {
                key = node_name + "->" + key;
                m_indirect_literals[key] = value;
            }
        }
        else
        {
            throw_rte_with_backtrace("Invalid line:\n----\n", line, "\n----\n for node ",
                                     node_name);
        }
    }

    // if any param_names are not set, throw an error
    for (uint32 i = 0; i < param_names.size(); ++i)
    {
        if (param_values[i].empty())
            throw_rte_with_backtrace("Parameter `", param_names[i], "` for node `", node_name,
                                     "` is not set");
    }
    return param_values;
}

// define NodeCreatorFunc for each node type
NodePtr<FloatT> create_input_node(std::istream& is, const std::string& name,
                                  NetworkBuilder& builder)
{
    string_vector param_names = {"batch batch_size", "height num_samples", "width row_vec_size"};
    string_vector param_values = builder.read_params(param_names, is, name);
    return new Input<FloatT>(builder.get_value<uint32>(param_values[0]),
                             builder.get_value<uint32>(param_values[1]),
                             builder.get_value<uint32>(param_values[2]), name);
}

NodePtr<FloatT> create_dropout_node(std::istream& is, const std::string& name,
                                    NetworkBuilder& builder)
{
    string_vector param_names = {"rate dropout_rate", "prev input input_node"};
    string_vector param_values = builder.read_params(param_names, is, name);
    return new Dropout<FloatT>(builder.get_value<float>(param_values[0]),
                               builder.get_node(param_values[1]), name);
}

NodePtr<FloatT> create_linear_node(std::istream& is, const std::string& name,
                                   NetworkBuilder& builder)
{
    string_vector param_names = {"out_dim output_dim", "prev input input_node", "bias use_bias",
                                 "activation activation_function"};
    string_vector param_values = builder.read_params(param_names, is, name);
    uint32 out_dim = builder.get_value<uint32>(param_values[0]);
    NodePtr<FloatT> prev = builder.get_node(param_values[1]);
    bool use_bias = builder.get_value<bool>(param_values[2]);
    std::string activation_function = param_values[3];

    if (activation_function == "relu")
        return new Linear<FloatT, Relu<FloatT>>(out_dim, prev, use_bias, name);
    else if (activation_function == "sigmoid")
        return new Linear<FloatT, Sigmoid<FloatT>>(out_dim, prev, use_bias, name);
    else if (activation_function == "tanh")
        return new Linear<FloatT, TanH<FloatT>>(out_dim, prev, use_bias, name);
    else if (activation_function == "identity")
        return new Linear<FloatT, IActivation<FloatT>>(out_dim, prev, use_bias, name);
    else if (activation_function == "leaky_relu" || activation_function == "leakyrelu")
        return new Linear<FloatT, LeakyRelu<FloatT>>(out_dim, prev, use_bias, name);

    throw_rte_with_backtrace("Unknown activation function: ", activation_function);
    return nullptr;
}

NodePtr<FloatT> create_self_attention_node(std::istream& is, const std::string& name,
                                           NetworkBuilder& builder)
{
    string_vector param_values = builder.read_params(
        {"out_size output_size out_dim output_dim", "prev input input_node", "bias use_bias"}, is,
        name);
    return new SelfAttention<FloatT>(builder.get_value<uint32>(param_values[0]),
                                     builder.get_node(param_values[1]),
                                     builder.get_value<bool>(param_values[2]), name);
}

NodePtr<FloatT> create_l2_loss_node(std::istream& is, const std::string& name,
                                    NetworkBuilder& builder)
{
    string_vector param_values = builder.read_params({"predictions", "target"}, is, name);
    auto predictions = builder.get_node(param_values[0]);
    auto target = builder.get_node(param_values[1]);
    return new L2Loss<FloatT>({predictions, target}, name);
}

NodePtr<FloatT> create_attention_node(std::istream& is, const std::string& name,
                                      NetworkBuilder& builder)
{
    string_vector param_values = builder.read_params(
        {"q_size q_output_size q_out_dim", "q_input q_prev q_input_node", "q_bias",
         "k_size k_output_size k_out_dim", "k_input k_prev k_input_node", "k_bias",
         "v_size v_output_size v_out_dim", "v_input v_prev v_input_node", "v_bias"},
        is, name);

    uint32 sizes[3];
    NodePtr<FloatT> prevs[3];
    bool biases[3];
    for (uint32 i = 0; i < 3; ++i)
    {
        sizes[i] = builder.get_value<uint32>(param_values[i * 3]);
        prevs[i] = builder.get_node(param_values[i * 3 + 1]);
        biases[i] = builder.get_value<bool>(param_values[i * 3 + 2]);
    }

    return new Attention<FloatT>({sizes[0], prevs[0], biases[0], "Q" + name},
                                 {sizes[1], prevs[1], biases[1], "K" + name},
                                 {sizes[2], prevs[2], biases[2], "V" + name}, name);
}
NetworkBuilder::NetworkBuilder(std::string network_desc_filename)
{
    m_node_creators["Input"] = create_input_node;
    m_node_creators["Dropout"] = create_dropout_node;
    m_node_creators["Linear"] = create_linear_node;
    m_node_creators["SelfAttention"] = create_self_attention_node;
    m_node_creators["L2Loss"] = create_l2_loss_node;
    m_node_creators["Attention"] = create_attention_node;

    std::ifstream network_desc_file(network_desc_filename);
    parse_network_desc(network_desc_file);
    network_desc_file.close();

    network_desc_file.open(network_desc_filename);
    m_network_desc_string = std::string(std::istreambuf_iterator<char>(network_desc_file),
                                        std::istreambuf_iterator<char>());
    network_desc_file.close();
}

void NetworkBuilder::parse_network_desc(std::istream& is)
{
    std::string line;
    while (std::getline(is, line))
    {
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();
        if (line.back() == '\n') line.pop_back();

        // check if the line is a key_value_pair
        if (auto key_value_pair = parse_key_value_pair(line))
        {
            auto [key, value] = *key_value_pair;
            if (m_node_creators.count(key))
            {
                if (m_nodes.count(value))
                    throw_rte_with_backtrace("Node with name `", value,
                                             "` is being redefined on line ", get_line_number(is));
                m_nodes[value] = m_node_creators[key](is, value, *this);

                LOG(GREEN, "Created node ", value, " with type ", key);
            }
            else if (key[0] == '$')
                m_literals[key] = value;
            else
                throw_rte_with_backtrace("Unknown key: ", key);
        }
    }

    for (const auto& [name, node] : m_nodes)
    {
        bool is_not_loss = dynamic_cast<Loss2Node<FloatT>*>(node) == nullptr;
        if (m_used_nodes.count(node) == 0 and is_not_loss)
            throw_rte_with_backtrace("Node `", name, "` is not used");
    }

    for (const auto& [name, value] : m_literals)
    {
        if (m_used_literals.count(name) == 0)
            throw_rte_with_backtrace("Literal `", name, "` is not used");
    }
}
