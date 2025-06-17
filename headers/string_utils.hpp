/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef STRING_UTILS_HPP
#define STRING_UTILS_HPP

#include <map>
#include <regex>
#include <string>
#include <tuple>
#include <typeinfo>
#include "utils.hpp"

inline std::vector<std::string> split_str(const std::string& line, char delimiter, char quote)
{
    std::vector<std::string> result;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.length(); ++i)
    {
        char c = line[i];
        if (c == quote)
            in_quotes = !in_quotes;
        else if (c == delimiter && !in_quotes)
        {
            result.push_back(field);
            field.clear();
        }
        else
            field += c;
    }

    // Add the last field
    if (!field.empty())
    {
        result.push_back(field);
    }
    return result;
}

// Remove everything after a '#'
inline std::string strip_comments(const std::string& str, std::string comment_str)
{
    auto comment_pos = str.find(comment_str);
    if (comment_pos != std::string::npos) return str.substr(0, comment_pos);
    return str;
}

inline std::string strip_whitespace(const std::string& str)
{
    return std::regex_replace(str, std::regex("^\\s+|\\s+$"), "");
}

// get a line from the stream, return the original line, the stripped line, and whether it is a
// comment-only line
inline std::tuple<std::string, std::string, bool> get_line_(std::istream& is,
                                                            std::string comment_str)
{
    std::string orig;
    std::getline(is, orig);
    auto line = strip_whitespace(orig);
    bool is_comment = line[0] == comment_str[0];
    line = strip_comments(line, comment_str);
    return std::make_tuple(orig, line, is_comment);
}

// parse a line of the form "key: value", expect no starting or trailing whitespace, and no comments
// ':' can be replaced with any other separator
inline Optional<std::pair<std::string, std::string>> parse_key_value_pair(std::string line,
                                                                          std::string separator)
{
    if (line.empty()) return {};

    auto colon_pos = line.find(separator);
    if (colon_pos == std::string::npos) return {};

    std::string key = line.substr(0, colon_pos);
    std::string value = line.substr(colon_pos + 1);
    // strip whitespace from key &&value
    key = strip_whitespace(key);
    value = strip_whitespace(value);
    if (value.empty()) return {};
    return std::make_pair(key, value);
}

template <typename T>  // limited version of stringstream >> operator
T string_to_type(const std::string& str)
{
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
    else if constexpr (std::is_same<T, std::string>::value)
        return str;
    else if constexpr (std::is_same<T, const char*>::value)
        return str.c_str();
    throw_rte_with_backtrace("Unsupported type: ", typeid(T).name());
}

// A structure to emulate Python's **kwargs
struct VarArgs
{
    std::map<std::string, std::string> args;

    VarArgs(std::initializer_list<std::pair<const char*, const char*>> args)
    {
        for (const auto& arg : args)
        {
            this->args[arg.first] = arg.second;
        }
    }

    VarArgs(std::istream& key_value_pair_stream)
    {
        std::string line;
        while (std::getline(key_value_pair_stream, line))
        {
            auto key_value_pair = parse_key_value_pair(line, ":");
            if (key_value_pair)
            {
                args[key_value_pair->first] = key_value_pair->second;
            }
        }
    }

    template <typename T>
    void set(const std::string& key, T value)
    {
        args[key] = value;
    }

    template <typename T>
    Optional<T> get(const std::string& key) const
    {
        auto it = args.find(key);
        if (it == args.end()) return Optional<T>();
        return Optional<T>(string_to_type<T>(it->second));
    }

    template <typename T>
    T get(const std::string& key, T default_value) const
    {
        auto it = args.find(key);
        if (it == args.end()) return default_value;
        return string_to_type<T>(it->second);
    }

    std::string get(const std::string& key, const std::string& default_value) const
    {
        auto it = args.find(key);
        if (it == args.end()) return default_value;
        return it->second;
    }
};

#endif
