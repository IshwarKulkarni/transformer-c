#ifndef NETWORK_BUILDER_HPP
#define NETWORK_BUILDER_HPP

#include "nodes/node.hpp"

struct NetworkBuilder;

using string_string_map = std::map<std::string, std::string>;
using string_vector = std::vector<std::string>;
using string_pair_vec = std::vector<std::pair<std::string, std::string>>;

using NodePtrMap = std::map<std::string, NodePtr<FloatT>>;
using LiteralMap = std::map<std::string, std::string>;
using NodeCreatorFunc = NodePtr<FloatT> (*)(std::istream& is, const std::string& name, NetworkBuilder& builder);

void initialize_node_creators();
string_pair_vec::iterator match_key_substr(const std::string& key, string_pair_vec& opt_params);
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

struct NetworkBuilder
{
    /* grammar for network description language:

        network_desc = node* | key_value_pair | node*

        Node :=  key_value_pair | newline | definition_lines | newline
        definition_lines := definition_line+
        definition_line := "\t" | key_value_pair | newline
        key_value_pair := key | ws | colon | ws | value
        key := <identifier>
        value := <literal>
        identifier := <alphanumeric>+
        literal := <alphanumeric>+
        ws := [" " | "\t"]+
        colon := ":"
    */

    private:
    LiteralMap m_literals;
    NodePtrMap m_nodes;
    std::set<std::string> m_used_literals;
    std::set<NodePtr<FloatT>> m_used_nodes;
    std::string m_network_desc_string;
    string_string_map m_indirect_literals;

    public:

    NetworkBuilder(std::string network_desc_filename);

    void parse_network_desc(std::istream& is);

    void read_params(std::istream& is, const std::string& node_name, string_pair_vec& key_vals);

    ~NetworkBuilder()
    {
        for (auto& [name, node] : m_nodes)
        {
            delete node;
        }
    }

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


    template <typename T>
    T parse_value(const std::string& str)
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

    template <typename T> // if name is in m_literals, return the literal as T, else parse param_value as T and return the parsed value    
    T get_value(const std::string& param_value)
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

    const NodePtrMap& get_nodes() const
    {
        return m_nodes;
    }

    const std::string& get_network_desc_string() const
    {
        return m_network_desc_string;
    }
};

#endif  // NETWORK_BUILDER_HPP
