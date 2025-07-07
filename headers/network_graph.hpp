/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef NETWORK_BUILDER_HPP
#define NETWORK_BUILDER_HPP

#include <pthread.h>
#include <algorithm>
#include "errors.hpp"
#include "nodes/node.hpp"
#include "string_utils.hpp"

const std::string None = "[None]";

struct NetworkGraph;
class NodeCreator
{
 public:
    virtual ~NodeCreator() = default;
    virtual NodePtr<FloatT> create(std::istream& is, const std::string& name,
                                   NetworkGraph& graph) = 0;
    void help(std::ostream& os) const
    {
        for (const auto& [param, default_value] : m_param_defaults)
        {
            std::string val_str =
                default_value == None ? "" : "(default val:\t" + default_value + ")";
            os << "  " << param << ": " << val_str << "\n";
        }
    }

    void populate_params(NetworkGraph& graph, std::istream& is, const std::string& node_name);
    const std::string NodeType;

 protected:
    // a map defining paramters to construct the node with their default values, is default is
    // NoneString, that param needs to be provided
    StringStringMap m_param_defaults;
    NodeCreator(const std::string& node_type, const StringStringMap& param_defaults,
                bool has_prev = true)
        : NodeType(node_type), m_param_defaults(param_defaults)
    {
        if (has_prev)
        {
            m_param_defaults["prev"] = "[None]";
        }
    }

    NodePtr<FloatT> get_prev_node(NetworkGraph& graph);
    template <typename T>
    T get_value(NetworkGraph& graph, const std::string& variable_name);
    StringVector get_param_names() const
    {
        StringVector names;
        for (const auto& [name, value] : m_param_defaults)
        {
            names.push_back(name);
        }
        return names;
    }

    std::string get_param_value(const std::string& variable_name) const
    {
        auto it = m_param_defaults.find(variable_name);
        if (it == m_param_defaults.end())
            throw_rte_with_backtrace(
                "Parameter ", variable_name, " not found in parameter list of for node ", NodeType,
                " it take one of the following values: ", join(get_param_names(), "\n"));
        return it->second;
    }
};

struct NodeCreatorMap
{
    static std::map<std::string, std::unique_ptr<NodeCreator>> m_node_creators;

    static void initialize();

    static NodeCreator* get(const std::string& name);
    static bool has(const std::string& name)
    {
        return m_node_creators.find(name) != m_node_creators.end();
    }

    static StringVector get_creator_names()
    {
        StringVector names;
        for (const auto& [name, creator] : m_node_creators)
        {
            names.push_back(name);
        }
        return names;
    }
};

/*
    NetworkGraph is a class that represents a directed, acyclic graph of nodes; A tree;
    It is used to create a network of nodes from a description of the network.
    It also provides logic for saving and loading the network to and from a file
    (saves it with description and weights)

    Incomplete Grammar for network description language:
    // incomplete because it's not context free (need to parse nested structures/nodes to define
    other nodes)

    NetworkGraph := node* | literal_def* | node* | comment*

    node := name_value_pair | newline | definition_lines | newline
    definition_lines := definition_line+
    definition_line := key_value_pair | newline
    name_value_pair := name | ws | colon | ws | value | comment
    name := <identifier>
    value := <literal>
    identifier := <alphanumeric>+
    literal := name | number | boolean | string
    number := numeral | numeral "." numeral
    boolean := "true" | "false" | "1" | "0" | "yes" | "no"
    string := <string_literal>
    string_literal := <string_literal_char>+ | "_" | <alphanumeric>+
    key_value_pair := ws* | key | ws | colon | ws | value | comment
    literal_def := literal_key ":" literal_key
    comment := "#" ws* | * | newline
*/

typedef std::pair<NodePtr<FloatT>, NodePtr<FloatT>> Edge;

using NodePtrMap = std::map<std::string, NodePtr<FloatT>>;
using LiteralMap = std::map<std::string, std::string>;
StringPairVec::iterator match_key_substr(const std::string& key, StringPairVec& opt_params);

struct NetworkGraph
{
 private:
    LiteralMap m_literals;
    NodePtrMap m_nodes;  // values are deleted in the destructor`
    std::vector<std::pair<NodePtr<FloatT>, std::string>>
        m_nodes_sorted;  // sorted by creation order
    std::set<std::string> m_used_literals;
    std::string m_network_desc_string;
    StringStringMap m_indirect_literals;

    static constexpr uint32 MAGIC_NUMBER = 0xBA560055;
    static constexpr uint32 VERSION_MAJOR = 0;
    static constexpr uint32 VERSION_MINOR = 1;
    static constexpr uint32 HEADER[4] = {MAGIC_NUMBER, VERSION_MAJOR, VERSION_MINOR, 0};
    static constexpr char TEXT_DELIM[] = "--------------------------------";

    NodePtr<FloatT> m_root_node = nullptr;

    void clear()
    {
        for (const auto& [name, node] : m_nodes)
        {
            delete node;
        }
        m_nodes.clear();
        m_literals.clear();
        m_used_literals.clear();
        m_indirect_literals.clear();
        m_root_node = nullptr;
    }

    void parse_network_desc();  // load network description from m_network_desc_string

    template <typename T>
    inline T parse_value(const std::string& str)
    {
        try
        {
            return string_to_type<T>(str);
        }
        catch (const std::invalid_argument& e)
        {
            throw_rte_with_backtrace("Invalid argument in parsing `", str,
                                     "` with function: ", e.what(),
                                     ". Is it a literal? They begin with `$`");
        }
        return T();
    }

 public:
    NetworkGraph() {}
    explicit NetworkGraph(std::string network_desc_filename);

    // Method added specifically for testing variable resolution
    template <typename T>
    void define_variable(const std::string& name, T value)
    {
        if (name.empty() || name[0] != '$')
        {
            throw_rte_with_backtrace("Variable name must start with '$': ", name);
        }
        std::stringstream ss;
        ss << value;
        m_literals[name] = ss.str();
    }

    void load_from_desc_stream(std::istream& is);

    bool attempt_load_weight_file(std::string filename);

    ~NetworkGraph() { clear(); }

    void read_params(std::istream& is, const std::string& node_name, StringStringMap& key_vals,
                     bool rewind = false);

    StringStringMap get_key_value_pairs(std::istream& is, const std::string& node_name);

    StringVector get_node_names()
    {
        StringVector node_names(m_nodes.size());
        std::transform(m_nodes.begin(), m_nodes.end(), node_names.begin(),
                       [](const auto& pair) { return pair.first; });
        return node_names;
    }

    NodePtr<FloatT> get_node(const std::string& name)
    {
        if (name.empty()) throw_rte_with_backtrace("Empty node name");
        auto it = m_nodes.find(name);
        if (it != m_nodes.end())
        {
            return it->second;
        }

        StringVector node_names = get_node_names();
        auto [closest_match, dist] = get_closest_match(node_names.begin(), node_names.end(), name);
        std::string perhaps = "";
        if (dist < 0.3) perhaps += YELLOW + closest_match + RESET + " perhaps?";
        LOG(YELLOW, "Available nodes: ");
        for (const auto& [node_name, node] : m_nodes)
        {
            LOG(YELLOW, node_name, " - ", node->type());
        }
        throw_rte_with_backtrace("Node `", name, "` undefined.", perhaps);
        return nullptr;
    }

    void print_nodes() const;        // print node names, shapes, and types
    void print_node_values() const;  // print node values as matrices

    template <typename NodeType>
    NodeType* get_typed_node(const std::string& name)
    {
        auto it = m_nodes.find(name);
        if (it != m_nodes.end())
        {
            auto node = dynamic_cast<NodeType*>(it->second);
            if (!node)
                throw_rte_with_backtrace("Node with name `", name, "` is not of type ",
                                         typeid(NodeType).name());
            return node;
        }
        auto node_names = get_node_names();
        auto [match, dist] = get_closest_match(node_names.begin(), node_names.end(), name);
        std::string perhaps = "";
        if (dist < 0.3) perhaps += YELLOW + match + RESET + " perhaps?";
        throw_rte_with_backtrace("Node with name `", name, "` is not defined.", perhaps);
        return nullptr;
    }

    void add_node_ptr(NodePtr<FloatT> node) { m_nodes[node->name] = node; }

    template <typename T>  // if name is in m_literals, return the literal as T, else parse
                           // param_value as T and return the parsed value
    inline T get_value(const std::string& param_value)
    {
        if (param_value.empty())
            throw_rte_with_backtrace("Empty string is not a valid parameter value");
        if (param_value[0] == '$')
        {
            auto it = m_literals.find(param_value);
            if (it == m_literals.end())
            {
                auto it = param_value.find("->");
                if (it == std::string::npos)  // not an indirect literal
                {
                    // print all literals
                    LOG(YELLOW, "Available literals: ");
                    for (const auto& [key, value] : m_literals)
                    {
                        LOG(YELLOW, key, " - ", value);
                    }
                    throw_rte_with_backtrace("Literal `", param_value, "` is not defined");
                }
                throw_rte_with_backtrace("Indirect literal `", param_value,
                                         "` cannot begin with `$`");
            }
            m_used_literals.insert(it->first);
            return parse_value<T>(it->second);
        }
        if (param_value.find("->") != std::string::npos)  // indirect literal
        {
            auto it = m_indirect_literals.find(param_value);
            if (it == m_indirect_literals.end())
            {
                auto [node_name, key] = split(param_value, "->");
                auto node_found = m_nodes.find(node_name);
                if (node_found == m_nodes.end())
                {
                    throw_rte_with_backtrace("Node `", node_name, "`, used in indirect literal `",
                                             param_value,
                                             "`, needs to be textually defined before being used "
                                             "in an indirect literal");
                }
                else
                {
                    LOG(YELLOW, "Available indirect literals: ");
                    for (auto& literal : m_indirect_literals)
                    {
                        LOG(YELLOW, literal.first, " - ", literal.second);
                    }
                    throw_rte_with_backtrace("Node `", node_name, "` does not have a key `", key,
                                             "`");
                }
            }
            return get_value<T>(it->second);
        }
        return parse_value<T>(param_value);
    }

    template <typename T>
    inline Optional<T> get_value_optionally(const std::string& param_value)
    {
        try
        {
            return get_value<T>(param_value);
        }
        catch (const std::invalid_argument& e)
        {
            return Optional<T>();
        }
    }

    NodePtr<FloatT> get_root_node() const;

    const std::string& get_network_desc_string() const { return m_network_desc_string; }

    // save the network description to a file, followed by nodes and their weights
    // the format is:
    // network_desc
    // ###########
    // node_name: node weights
    // node_name: node weights
    // Nodes appear in sorted order of their names
    void save_network(const std::string& filename) const;

    // TODO: handle multiple root nodes
    inline void write_dotviz(std::string filename) { write_dotviz(filename, get_root_node()); }

    static void write_dotviz(std::string filename, const NodePtr<FloatT> node);

    const NodePtrMap& get_nodes() const { return m_nodes; }

    void print_nodes_sorted() const
    {
        cudaErrCheck(cudaDeviceSynchronize());
        for (const auto& [node, key] : m_nodes_sorted)
            LOG(RED, node->name, " - ", node->type(), "\n", *node, "\n");
    }

    void set_all_is_training(bool is_training)
    {
        NodeBase::set_all_is_training(is_training);
        ParameterBase::set_all_is_training(is_training);
    }
};

#endif  // NETWORK_BUILDER_HPP
