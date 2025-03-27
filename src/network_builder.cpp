// network_builder.cpp, 2025-03-24, 10:00
// uses a simple network description language to build networks, with nodes of types of child
// classes of Node

#include "network_builder.hpp"
#include <fstream>
#include <istream>
#include <regex>
#include <sstream>
#include "nodes/loss.hpp"
#include "nodes/node.hpp"
#include "utils.hpp"

std::string strip_whitespace(const std::string& str)
{
    return std::regex_replace(str, std::regex("^\\s+|\\s+$"), "");
}

std::string strip_comments(const std::string& str)
{
    auto comment_pos = str.find("#");
    if (comment_pos != std::string::npos) return str.substr(0, comment_pos);
    return str;
}

// get a line from the stream, return the original line, the stripped line, the line number, and
// whether it is a comment stripped line is the line with whitespace removed and comments removed
std::tuple<std::string, std::string, uint32, bool> get_line_(std::istream& is)
{
    static uint32 line_number = 0;
    std::string orig;
    std::getline(is, orig);
    line_number++;
    auto line = strip_whitespace(orig);
    bool is_comment = line[0] == '#';
    line = strip_comments(orig);
    return std::make_tuple(orig, line, line_number, is_comment);
}

// parse a line of the form "key: value", expect no starting or trailing whitespace, and no comments
static Optional<std::pair<std::string, std::string>> parse_key_value_pair(std::string line)
{
    if (line.empty()) return {};

    auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) return {};

    std::string key = line.substr(0, colon_pos);
    std::string value = line.substr(colon_pos + 1);
    // strip whitespace from key and value
    key = strip_whitespace(key);
    value = strip_whitespace(value);
    if (value.empty()) return {};
    return std::make_pair(key, value);
}

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
string_pair_vec::iterator match_key_substr(const std::string& key, string_pair_vec& key_vals)
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
void NetworkBuilder::read_params(std::istream& is, const std::string& node_name,
                                 string_pair_vec& key_vals)
{
    string_pair_vec key_value_pairs;  // all key value pairs in block

    while (is)
    {
        auto [orig, line, line_number, is_comment] = get_line_(is);
        if (is_comment) continue;
        if (line.empty()) break;
        auto key_value_pair = parse_key_value_pair(line);
        if (!key_value_pair)
            throw_rte_with_backtrace("Invalid line:\n----\n", orig, "\n----\n for node`", node_name,
                                     "` not a key-value pair; near line ", line_number);
        key_value_pairs.push_back(*key_value_pair);
    }

    for (const auto& [key, value] : key_value_pairs)
    {
        if (key.empty())
            throw_rte_with_backtrace("Parameter `", key, "` in node `", node_name,
                                     "` is not defined");
        auto it = match_key_substr(key, key_vals);
        if (it == key_vals.end())
            throw_rte_with_backtrace("Parameter `", key, "` in node `", node_name, "` unknown");
        it->second = value;
        m_indirect_literals[node_name + "->" + key] = value;
    }

    for (const auto& [key, value] : key_vals)
    {
        if (value.empty())
            throw_rte_with_backtrace("Parameter `", key, "` in node `", node_name,
                                     "` is not defined");
    }
}

NetworkBuilder::NetworkBuilder(std::string network_desc_filename)
{
    initialize_node_creators();  // Initialize all node creator functions
    if (!attempt_load_weight_file(network_desc_filename))
    {
        std::ifstream network_desc_file(network_desc_filename);
        m_network_desc_string = std::string(std::istreambuf_iterator<char>(network_desc_file),
                                            std::istreambuf_iterator<char>());
        network_desc_file.close();
        parse_network_desc();
    }
}

bool NetworkBuilder::attempt_load_weight_file(std::string filename)
{
    std::ifstream file_in(filename, std::ios::in | std::ios::binary);
    if (!file_in) throw_rte_with_backtrace("File `", filename, "` cannot be opened");

    uint32 header[5];
    file_in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (header[0] != MAGIC_NUMBER or header[3] != 0)
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

void NetworkBuilder::parse_network_desc()
{
    std::stringstream is(m_network_desc_string);
    while (is)
    {
        auto [orig, line, line_number, is_comment] = get_line_(is);
        if (is_comment) continue;

        // check if the line is a key_value_pair
        if (auto key_value_pair = parse_key_value_pair(line))
        {
            auto [key, value] = *key_value_pair;
            if (NodeCreatorMap::has(key))
            {
                if (m_nodes.count(value))
                    throw_rte_with_backtrace("Node with name `", value,
                                             "` is being redefined on line ", line_number);
                m_nodes[value] = NodeCreatorMap::get(key)(is, value, *this);

                LOG(GREEN, "Created node ", value, " with type ", key);
            }
            else if (key[0] == '$')
            {
                if (!is_valid_literal(key))
                    throw_rte_with_backtrace("Literal `", key, "` is not valid literal");
                m_literals[key] = value;
            }
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
        if (m_used_literals.count(name) == 0) LOG(YELLOW, "Literal `", name, "` is not used");
    }
}

void NetworkBuilder::save_network(const std::string& filename) const
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
    for (auto& [name, node] : m_nodes) node_names.push_back(name);
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