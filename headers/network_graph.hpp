#ifndef NETWORK_BUILDER_HPP
#define NETWORK_BUILDER_HPP

#include <pthread.h>
#include "errors.hpp"
#include "nodes/node.hpp"

/*
    NetworkGraph is a class that represents a directed, acyclic graph of nodes; A tree;
    It is used to create a network of nodes from a description of the network.
    It also provides logic for saving and loading the network to and from a file
    (saves it with description and weights)

    Incomplete Grammar for network description language: 
    // incomplete because it's not context free (need to parse nested structures/nodes to define other nodes)

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

using StringStringMap = std::map<std::string, std::string>;
using StringVector = std::vector<std::string>;
using StringPairVec = std::vector<std::pair<std::string, std::string>>;

using NodePtrMap = std::map<std::string, NodePtr<FloatT>>;
using LiteralMap = std::map<std::string, std::string>;
using NodeCreatorFunc = NodePtr<FloatT> (*)(std::istream& is, const std::string& name, NetworkGraph& builder);

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
        if(m_node_creators.find(name) == m_node_creators.end())
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
    NodePtrMap m_nodes;
    std::set<std::string> m_used_literals;
    std::set<NodePtr<FloatT>> m_used_nodes;
    std::string m_network_desc_string;
    StringStringMap m_indirect_literals;

    static constexpr uint32 MAGIC_NUMBER = 0xBA560055;
    static constexpr uint32 VERSION_MAJOR = 0;
    static constexpr uint32 VERSION_MINOR = 1;
    static constexpr uint32 HEADER[4] = {MAGIC_NUMBER, VERSION_MAJOR, VERSION_MINOR, 0};
    static constexpr char TEXT_DELIM[] = "--------------------------------";

    public:

    NetworkGraph() {};
    explicit NetworkGraph(std::string network_desc_filename);

    void load_from_desc_stream(std::istream& is);

    ~NetworkGraph()
    {
        for (const auto& [name, node] : m_nodes)
        {
            delete node;
        }
    }

    bool attempt_load_weight_file(std::string filename);

    void parse_network_desc(); // load network description from m_network_desc_string

    void read_params(std::istream& is, const std::string& node_name, StringPairVec& key_vals);

    NodePtr<FloatT> get_node(const std::string& name)
    {
        auto it = m_nodes.find(name);
        if (it != m_nodes.end())
        {
            m_used_nodes.insert(it->second);
            return it->second;
        }
        throw_rte_with_backtrace("Node with name `", name, "` is not defined");
        return nullptr;
    }

    template<typename NodeType>
    NodeType* get_typed_node(const std::string& name)
    {
       auto it = m_nodes.find(name);
        if (it != m_nodes.end())
        {
            auto node = dynamic_cast<NodeType*>(it->second);
            if(!node)
                throw_rte_with_backtrace("Node with name `", name, "` is not of type ", typeid(NodeType).name());
            return node;
        }
        throw_rte_with_backtrace("Node with name `", name, "` is not defined");
        return nullptr;
    }   

    template <typename T>
    inline T parse_value(const std::string& str)
    {
        try {
        if constexpr (std::is_same<T, float>::value)
            return std::stof(str);
        else if constexpr (std::is_same<T, int>::value)
            return std::stoi(str);
        else if constexpr (std::is_same<T, double>::value)
            return std::stod(str);
        else if constexpr (std::is_same<T, bool>::value)
            return str == "true" || str == "1" || str == "yes" || str == "y";
        else if constexpr (std::is_same<T, uint32>::value)
            return std::stoul(str);
        else if constexpr (std::is_same<T, uint64>::value)
                return std::stoull(str);
        else 
            throw_rte_with_backtrace("Unsupported type: ", typeid(T).name());
        } catch (const std::invalid_argument& e) {
            throw_rte_with_backtrace("Invalid argument in parsing `", str, "` with function: ",
                                     e.what(), ". Is it a literal? They begin with `$`");
        }
    }

    void train();

    template <typename T> // if name is in m_literals, return the literal as T, else parse param_value as T and return the parsed value    
    inline T get_value(const std::string& param_value)
    {
        if(param_value[0] == '$')
        {
            auto it = m_literals.find(param_value);
            if (it == m_literals.end())
            {
                auto it = param_value.find("->");
                if (it == std::string::npos)
                {
                    throw_rte_with_backtrace("Literal `", param_value, "` is not defined");
                    // print all literals
                    for (const auto& [key, value] : m_literals)
                    {
                        LOG(YELLOW, key, " - ", value);
                    }
                }
                throw_rte_with_backtrace("Indirect literal `", param_value, "` cannot begin with `$`");
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
                    throw_rte_with_backtrace("Node `", node_name, "`, used in indirect literal `", param_value,
                    "` it needs to be textually defined before being used in an indirect literal");
                }
                else
                {
                    throw_rte_with_backtrace("Node `", node_name, "` does not have a key `", key, "`");
                }
            }
            return get_value<T>(it->second);
        }
        return parse_value<T>(param_value);
    }

    NodePtr<FloatT> get_root_node() const
    {
        std::set<NodePtr<FloatT>> all_nodes;
        for (const auto& [name, node] : m_nodes)
        {
            all_nodes.insert(node);
        }
        for (const auto& [name, node] : m_nodes)
        {
            for (const auto& prev_node : node->prev_nodes)
            {
                all_nodes.erase(prev_node);
            }
        }
        if(all_nodes.empty())
            throw_rte_with_backtrace("There's Loop in the network");
        if(all_nodes.size() > 1)
            throw_rte_with_backtrace("There's more than one root node in the network");
        return *all_nodes.begin();
    }

    const std::string& get_network_desc_string() const
    {
        return m_network_desc_string;
    }

    // save the network description to a file, followed by nodes and their weights
    // the format is:
    // network_desc 
    // ###########
    // node_name: node weights
    // node_name: node weights
    // Nodes appear in sorted order of their names
    void save_network(const std::string& filename) const;

    inline void to_dotviz_file(std::string filename)
    {
        to_dotviz_file(filename, get_root_node());
    }

    static void to_dotviz_file(std::string filename, const NodePtr<FloatT> node);

    const NodePtrMap& get_nodes() const
    {
        return m_nodes;
    }
};

#endif  // NETWORK_BUILDER_HPP
