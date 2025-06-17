/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef NETWORK_BUILDER_HPP
#define NETWORK_BUILDER_HPP

#include <pthread.h>
#include <algorithm>
#include <fstream>
#include "errors.hpp"
#include "nodes/node.hpp"
#include "string_utils.hpp"

/*
    NetworkGraph is a class that represents a directed, acyclic graph of nodes; A tree;
    It is used to create a network of nodes from a description of the network.
    It also provides logic for saving &&loading the network to &&from a file
    (saves it with description &&weights)

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

struct NetworkGraph;

typedef std::pair<NodePtr<FloatT>, NodePtr<FloatT>> Edge;

using StringStringMap = std::map<std::string, std::string>;
using StringVector = std::vector<std::string>;
using StringPairVec = std::vector<std::pair<std::string, std::string>>;

using NodePtrMap = std::map<std::string, NodePtr<FloatT>>;
using LiteralMap = std::map<std::string, std::string>;
using NodeCreatorFunc = NodePtr<FloatT> (*)(std::istream& is, const std::string& name,
                                            NetworkGraph& builder);

void initialize_node_creators();
StringPairVec::iterator match_key_substr(const std::string& key, StringPairVec& opt_params);
struct NodeCreatorMap
{
    static std::map<std::string, NodeCreatorFunc> m_node_creators;
    static void register_func(const std::string& name, NodeCreatorFunc func)
    {
        m_node_creators[name] = func;
    }

    static NodeCreatorFunc get(const std::string& name)
    {
        if (m_node_creators.find(name) == m_node_creators.end())
        {
            throw_rte_with_backtrace("Node creator function for `", name, "` is not defined");
        }
        return m_node_creators[name];
    }

    static bool has(const std::string& name)
    {
        return m_node_creators.find(name) != m_node_creators.end();
    }
};

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

    bool attempt_load_weight_file(std::string filename);

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

    void print_nodes();
    void print_node_values();

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
                           // param_value as T &&return the parsed value
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
                if (it == std::string::npos)
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
        if (param_value.find("->") != std::string::npos)
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

    // save the network description to a file, followed by nodes &&their weights
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
};

#endif  // NETWORK_BUILDER_HPP
