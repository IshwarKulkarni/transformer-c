/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include "network_graph.hpp"
#include <fstream>
#include <istream>
#include <regex>
#include <sstream>
#include "logger.hpp"
#include "nodes/loss.hpp"
#include "nodes/node.hpp"
#include "string_utils.hpp"
#include "utils.hpp"
/*
    This file contains functions and methods related to parsing the network description file.
    Loading and saving the network to a file, writing DOT Viz file etc. The "graph" related
    methods are in header file.
*/

inline bool is_alpha_numeric(char c) { return std::isalnum(c) || c == '_'; }

// literal can only contain alphanumeric characters, underscores, and cannot begin with a number
inline bool is_valid_literal(const std::string& str)
{
    if (str.length() < 2) return false;
    if (str[0] != '$') return false;
    // cannot begin with a number
    if (isdigit(str[1])) return false;
    for (uint32 i = 1; i < str.length(); i++)
    {
        if (!is_alpha_numeric(str[i])) return false;
    }
    return true;
}

inline uint32 get_line_number(std::istream& is)
{
    auto pos = is.tellg();
    is.seekg(0, std::ios::beg);
    // count the number of newlines until pos
    uint32 currentPosition = 0;
    uint32 newlineCount = 0;
    char c;
    while (is.get(c) && currentPosition++ < pos)
        if (c == '\n') newlineCount++;
    is.seekg(pos);
    return newlineCount;
}

// given a string, return the iterator to the first key in key_vals that is a super-string(??) of
// the given string
StringPairVec::iterator match_key_substr(const std::string& key, StringPairVec& key_vals)
{
    for (auto it = key_vals.begin(); it != key_vals.end(); ++it)
    {
        auto& [o_key, o_val] = *it;
        if (o_key.find(key) != std::string::npos) return it;
    }
    return key_vals.end();
}

// read lines from is until end of block, get key value pairs, see if key is a substring of
// key_vals[i].first for some i, if so, set key_vals[i].second to the value, if no such key_vals[i]
// is found, throw an error if an empty line is encountered before all key values are set, throw an
// error end of block is an empty line.
// is: input stream
// node_name: name of the node
// key_vals: map to fill  values into, where key is the defined parameter name
// rewind: if true, reset the stream to the position it was at before the call (so multiple
// read_params can be called on the same block);
void NetworkGraph::read_params(std::istream& is, const std::string& node_name,
                               StringStringMap& key_vals, bool rewind)
{
    auto pos = is.tellg();

    auto key_value_pairs = get_key_value_pairs(is, node_name);

    for (const auto& [key, value] : key_value_pairs)
    {
        auto it = key_vals.find(key);
        if (it != key_vals.end())
        {
            it->second = value;
            m_indirect_literals[node_name + "->" + key] = value;
        }
    }
    if (rewind) is.seekg(pos);
}

StringStringMap NetworkGraph::get_key_value_pairs(std::istream& is, const std::string& node_name)
{
    StringStringMap key_value_pairs;  // all key value pairs in block
    while (is)
    {
        auto [orig, line, is_comment] = get_line_(is, "#");
        if (line.empty() && !is_comment) break;  // end of block
        if (is_comment) continue;                // comment
        auto key_value_pair = parse_key_value_pair(line, ":");
        if (!key_value_pair)
            throw_rte_with_backtrace("Invalid line:\n----\n", orig, "\n----\n for node`", node_name,
                                     "` not a key-value pair; near line:\n\t ", YELLOW, orig);
        key_value_pairs[key_value_pair->first] = key_value_pair->second;
    }
    // add to m_indirect_literals
    for (auto& [key, value] : key_value_pairs)
    {
        m_indirect_literals[node_name + "->" + key] = value;
    }
    return key_value_pairs;
}

NetworkGraph::NetworkGraph(std::string filename)
{
    if (!attempt_load_weight_file(filename))
    {
        std::ifstream network_desc_file(filename);
        load_from_desc_stream(network_desc_file);
    }
    if (m_nodes.empty())
        throw_rte_with_backtrace("No nodes created in network description file `", filename, "`");
}

void NetworkGraph::load_from_desc_stream(std::istream& in_stream)
{
    if (!in_stream) throw_rte_with_backtrace("File cannot be opened");
    m_network_desc_string =
        std::string(std::istreambuf_iterator<char>(in_stream), std::istreambuf_iterator<char>());
    parse_network_desc();
}

bool NetworkGraph::attempt_load_weight_file(std::string filename)
{
    std::ifstream file_in(filename, std::ios::in | std::ios::binary);
    if (!file_in) throw_rte_with_backtrace("File `", filename, "` cannot be opened");

    uint32 header[5];
    file_in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (header[0] != MAGIC_NUMBER || header[3] != 0)
    {
        return false;
    }
    if (header[1] != VERSION_MAJOR)
    {
        throw_rte_with_backtrace("Version mismatch: ", header[1], " != ", VERSION_MAJOR);
    }
    if (header[2] != VERSION_MINOR)
    {
        LOG(YELLOW, "Version mismatch: ", header[2], " != ", VERSION_MINOR,
            " will attempt to load");
    }
    uint32 text_length = header[4];
    m_network_desc_string = std::string(text_length, '\0');
    file_in.read(&m_network_desc_string[0], text_length);
    parse_network_desc();
    LOG(GREEN, "Loaded network description from file `", filename, "`");

    // load weights as written in save_network

    std::set<std::string> all_node_names;
    for (auto& [name, _] : m_nodes) all_node_names.insert(name);

    while (file_in)
    {
        uint32 name_length;
        file_in.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        if (name_length == 0) break;
        std::string node_name(name_length, '\0');
        file_in.read(&node_name[0], name_length);
        m_nodes.at(node_name)->load_weights(file_in);
        all_node_names.erase(node_name);
    }

    for (auto node_name : all_node_names)
        throw_rte_with_backtrace("`", node_name, "` could not be located in the saved file");
    return true;
}

void NetworkGraph::parse_network_desc()
{
    initialize_node_creators();
    std::stringstream is(m_network_desc_string);

    // clear all internal data &&reset the graph
    if (!m_nodes.empty())
    {
        LOG(RED, "Clearing network graph");
        clear();
    }
    while (is)
    {
        auto [orig, line, is_comment] = get_line_(is, "#");
        if (line.empty() || is_comment) continue;
        if (line == TEXT_DELIM) break;

        // check if the line is a key_value_pair
        if (auto key_value_pair = parse_key_value_pair(line, ":"))
        {
            auto [key, value] = *key_value_pair;
            if (NodeCreatorMap::has(key))
            {
                if (m_nodes.count(value))
                    throw_rte_with_backtrace("Node with name `", value,
                                             "` is being redefined on line:\n\t ", YELLOW, orig);
                try
                {
                    auto* node = NodeCreatorMap::get(key)(is, value, *this);
                    m_nodes[value] = node;
                    m_nodes_sorted.push_back(std::make_pair(node, value));
                }
                catch (const std::exception& e)
                {
                    LOG(RED, "Parsing error on line:\n\t", YELLOW, orig, RESET);
                    throw_rte_with_backtrace("Error creating node ", value);
                }
            }
            else if (key[0] == '$')
            {
                if (!is_valid_literal(key))
                    throw_rte_with_backtrace("Literal `", key, "` is not valid literal");
                m_literals[key] = value;
            }
            else
                throw_rte_with_backtrace("Unknown key: `", key, "`");
        }
    }

    for (const auto& [name, value] : m_literals)
    {
        if (m_used_literals.count(name) == 0) LOG(YELLOW, "Literal `", name, "` is not used");
    }

    this->m_root_node = get_root_node();
}

void NetworkGraph::save_network(const std::string& filename) const
{
    std::stringstream text;
    text << m_network_desc_string << "\n";
    text << TEXT_DELIM << "\n";

    std::ofstream file_out(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    uint32 text_length = text.str().length();
    file_out.write(reinterpret_cast<const char*>(HEADER), sizeof(HEADER));  // 4 uint32s : 16 bytes
    file_out.write(reinterpret_cast<const char*>(&text_length),
                   sizeof(text_length));  // 1 uint32 : 4 bytes
    file_out.write(text.str().c_str(), text_length);

    std::vector<std::string> node_names;
    for (auto& [name, _] : m_nodes) node_names.push_back(name);
    std::sort(node_names.begin(), node_names.end());

    for (auto& name : node_names)
    {
        uint32 name_length = name.length();
        file_out.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        file_out.write(name.c_str(), name_length);
        m_nodes.at(name)->save_weights(file_out);
    }
    uint32 zero = 0;
    file_out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));

    file_out.close();
}

void NetworkGraph::write_dotviz(std::string filename, const NodePtr<FloatT> node)
{
    NodePtrVec<FloatT> nodes;

    nodes.push_back(node);
    std::set<std::string> strings_reps;
    auto make_edge = [](NodePtr<FloatT> a, NodePtr<FloatT> b, float32 weight = 3.f,
                        const std::string& lbl = "") {
        char edge_buffer[256];
        snprintf(edge_buffer, 256, "%d -> %d [label=\"%dx%d %s\" weight=%2.1f]", a->id, b->id,
                 a->shape[1], a->shape[0], lbl.c_str(), weight);
        return std::string(edge_buffer);
    };

    while (!nodes.empty())
    {
        auto* n = nodes.back();
        nodes.pop_back();
        for (auto* p : n->prev_nodes)
        {
            strings_reps.insert(make_edge(p, n, 3.f));
            nodes.push_back(p);
            auto deps = p->get_dependencies();
            for (auto* d : deps)
            {
                strings_reps.insert(make_edge(d, p, 3.f));
                nodes.push_back(d);
            }
        }

        auto* terminal = n->get_terminal_node();
        if (terminal)
        {
            nodes.push_back(terminal);
            std::string term_edge = make_edge(n, terminal, 1.f, "\n-is-\n");
            term_edge = "\nedge [style=dotted arrowhead=none]\n" + term_edge +
                        "\nedge [style=normal arrowhead=normal];\n";
            strings_reps.insert(term_edge);
        }
        strings_reps.insert(std::to_string(n->id) + n->dot_repr());
    }
    std::ofstream os(filename);
    os << "digraph G {\n compound=true;\n";
    std::copy(strings_reps.begin(), strings_reps.end(),
              std::ostream_iterator<std::string>(os, "\n"));

    os << '}' << std::endl;
}

NodePtr<FloatT> NetworkGraph::get_root_node() const
{
    if (this->m_root_node)
    {
        return this->m_root_node;
    }

    if (m_nodes.empty())
    {
        throw_rte_with_backtrace("No nodes defined in the network");
    }

    if (m_nodes.size() == 1)
    {
        return m_nodes.begin()->second;
    }

    std::vector<Edge> forward_edges;

    for (const auto& [name, node] : m_nodes)
    {
        for (const auto& prev_node : node->get_dependencies())
        {
            forward_edges.push_back({prev_node, node});
        }
    }

    std::set<NodePtr<FloatT>> all_nodes;
    for (const auto& [name, node] : m_nodes) all_nodes.insert(node);

    for (const auto& edge : forward_edges) all_nodes.erase(edge.first);

    if (all_nodes.empty()) throw_rte_with_backtrace("There's Loop in the network");

    if (all_nodes.size() > 1)
    {
        for (const auto& node : all_nodes)
        {
            LOG(YELLOW, node->name);
        }
        throw_rte_with_backtrace("There's more than one root node in the network");
    }

    return *all_nodes.begin();
}

void NetworkGraph::print_nodes()
{
    uint32 total_param_count = 0;
    char buffer[36];
    setlocale(LC_NUMERIC, "");
    for (const auto& [node, key] : m_nodes_sorted)
    {
        uint32 id = node->id;
        snprintf(buffer, sizeof(buffer), "%5d", id);
        std::string id_str = std::string(buffer);

        snprintf(buffer, 21, "%20s", key.c_str());
        std::string key_str = std::string(buffer);

        snprintf(buffer, sizeof(buffer), "%-30s", node->type().c_str());
        std::string type_str = std::string(buffer);

        uint32 param_count = node->param_count();
        snprintf(buffer, sizeof(buffer), " |%'10d", param_count);
        std::string param_count_str =
            param_count ? std::string(buffer) : std::string(" |") + std::string(10, ' ');

        LOG(RED, id_str, param_count_str, key_str, ": ", type_str);
        total_param_count += param_count;
    }
    snprintf(buffer, sizeof(buffer), "Total |%'11d", total_param_count);
    LOG(RED, buffer);
}

void NetworkGraph::print_node_values()
{
    cudaErrCheck(cudaDeviceSynchronize());
    for (const auto& [node, key] : m_nodes_sorted)
    {
        LOG(key, ":\n", *node);
    }
}
