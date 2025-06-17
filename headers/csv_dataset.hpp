/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include "dataset.hpp"
#include "logger.hpp"
#include "string_utils.hpp"
#include "types"

struct CSV
{
    const char delimiter;
    const char quote;
    const bool has_header;

    CSV(std::string filename, bool has_header = true, char delimiter = ',', char quote = '"',
        uint32 max_samples = UINT32_MAX);

    const std::vector<std::string>& get_column_names() const { return header; }

    const std::string& get_column_name(uint32 col) const { return header[col]; }

    uint32 get_column_index(const std::string& col) const
    {
        auto it = header_to_index.find(col);
        if (it == header_to_index.end())
        {
            auto headers = get_column_names();
            auto [match, dist] = get_closest_match(headers.begin(), headers.end(), col);
            std::string perhaps = dist < 0.5 ? "`" + match + "` perhaps?" : "";
            throw_rte_with_backtrace("Column ", col, " not found in the csv file, did you mean ",
                                     YELLOW, perhaps, RESET, "?");
        }
        return it->second;
    }

    std::string row_str(uint32 row) const
    {
        if (row >= data.size())
        {
            throw_rte_with_backtrace("Row ", row, " is out of bounds");
        }
        std::string result;
        for (uint32 i = 0; i < header.size(); ++i)
        {
            result += header[i] + ":\t" + data[row][i] + "\n";
        }
        return result;
    }

    const std::string& operator()(uint32 col, uint32 row) const
    {
        if (row >= data.size())
        {
            throw_rte_with_backtrace("Row ", row, " is out of bounds");
        }
        if (col >= header.size())
        {
            throw_rte_with_backtrace("Column ", col, " is out of bounds");
        }
        return data[row][col];
    }

    const std::string& operator()(const std::string& col, uint32 row) const
    {
        auto it = header_to_index.find(col);
        if (it == header_to_index.end())
        {
            throw_rte_with_backtrace("Column ", col, " not found in the csv file");
        }
        return (*this)(it->second, row);
    }

    uint32 num_columns() const { return header.size(); }

    uint32 num_rows() const { return data.size(); }

    uint32 get_header_index(const std::string& col) const
    {
        auto it = header_to_index.find(col);
        if (it == header_to_index.end())
        {
            throw_rte_with_backtrace("Column ", col, " not found in the csv file");
        }
        return it->second;
    }

    void print_counts(const std::string& col);

    // print histogram of the column, using "*"s to represent the count ,
    template <typename T>
    void print_stats(const std::string& col, Optional<uint32> num_bins = 10);

 private:
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> data;
    std::map<std::string, uint32> header_to_index;
};

struct StrsToFloats
{
    std::vector<FloatT> operator()(const std::vector<std::string>& strs) const
    {
        std::vector<FloatT> floats;
        for (const auto& str : strs)
        {
            floats.push_back(std::stof(str));
        }
        return floats;
    }
};

struct StrsToOneHot
{
    uint32 m_classes;

    std::vector<FloatT> operator()(const std::vector<std::string>& strs) const
    {
        uint32 idx = std::stoi(strs[0]);
        std::vector<FloatT> one_hot(m_classes, 0.0f);
        if (idx >= m_classes)
        {
            throw_rte_with_backtrace("Index ", idx, " is out of bounds for one-hot encoding with ",
                                     m_classes, " classes");
        }
        one_hot[idx] = 1.0f;
        return one_hot;
    }
};

struct CSVDataset : public Dataset
{
    CSVDataset(std::string csv_filename, DataMode mode, Input<FloatT>* features_node,
               Input<FloatT>* target_node, const VarArgs& args);

    template <typename FeatureParser, typename TargetParser>
    void parse(const FeatureParser& feature_parser, const TargetParser& target_parser);

    // normalize features in place
    void normalize();
    void load(uint32 idx) override;

    bool is_one_hot() const { return m_one_hot_classes > 0; }

 private:
    std::set<uint32> m_tgt_cols;
    std::set<uint32> m_feat_cols;

    std::unique_ptr<CSV> m_csv;

    NodePtr<FloatT> m_features_node;
    NodePtr<FloatT> m_target_node;

    std::vector<std::vector<FloatT>> m_labels;
    std::vector<std::vector<FloatT>> m_features;

    std::vector<uint32> m_indices;

    uint32 m_one_hot_classes;
    bool m_do_normalize;
};
