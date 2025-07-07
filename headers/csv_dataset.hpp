#ifndef CSV_DATASET_HPP
#define CSV_DATASET_HPP
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
    void print_stats(const std::string& col, Optional<uint32> num_bins = 10);

 private:
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> data;
    std::map<std::string, uint32> header_to_index;
};

struct StrsToFloats
{
    std::vector<FloatT> operator()(const std::vector<std::string>& strs, uint32) const
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

    inline std::vector<FloatT> operator()(const std::vector<std::string>& strs, uint32) const
    {
        uint32 idx = std::stoi(strs[0]);
        std::vector<FloatT> one_hot(m_classes, 0.f);
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
    // constructor that calls the other constructor, and performs some size checks
    CSVDataset(std::string csv_filename, DataMode mode, Input<FloatT>* features_node,
               Input<FloatT>* target_node, const VarArgs& args);

    template <typename FeatureParser, typename TargetParser>
    void parse(FeatureParser& feature_parser, TargetParser& target_parser);

    // normalize features in place
    void normalize();
    void load(uint32 idx) override;

    bool is_one_hot() const { return m_one_hot_classes > 0; }

    ~CSVDataset() {}

    virtual void print_last_batch() const override
    {
        for (uint32 i = 0; i < m_features_node->batch(); i++)
        {
            uint32 csv_idx = m_last_load_indices[i];
            LOG(RED, i, " : ", csv_idx, " : ", m_csv->row_str(csv_idx));
        }
    }

 protected:
    // constructor that calls the other constructor, and performs some size checks
    CSVDataset(std::string csv_filename, DataMode mode, Input<FloatT>* features_node,
               Input<FloatT>* target_node, bool has_header, char delimiter, char quote,
               uint32 max_samples, uint32 one_hot_classes, const std::string& feature_col,
               const std::string& target_col);

    NodePtr<FloatT> m_features_node;
    NodePtr<FloatT> m_target_node;

    std::vector<std::vector<FloatT>> m_labels;
    std::vector<std::vector<FloatT>> m_features;

    uint32 m_one_hot_classes;

 private:
    std::set<uint32> m_tgt_cols;
    std::set<uint32> m_feat_cols;

    std::unique_ptr<CSV> m_csv;
    bool m_do_normalize;
};

template <typename FeatureParser, typename TargetParser>
void CSVDataset::parse(FeatureParser& feature_parser, TargetParser& target_parser)
{
    m_features.clear();
    m_labels.clear();

    std::vector<uint32> feat_cols;
    std::vector<uint32> tgt_cols;  // to preserve order of columns.
    std::string feat_col_names;
    std::string tgt_col_names;
    for (uint32 col = 0; col < m_csv->num_columns(); ++col)
    {
        if (m_feat_cols.find(col) != m_feat_cols.end())
        {
            feat_cols.push_back(col);
            feat_col_names +=
                m_csv->get_column_name(col) + (col < m_feat_cols.size() - 1 ? ", " : "");
        }
        if (m_tgt_cols.find(col) != m_tgt_cols.end())
        {
            tgt_cols.push_back(col);
            tgt_col_names +=
                m_csv->get_column_name(col) + (col < m_tgt_cols.size() - 1 ? ", " : "");
        }
    }

    if (mode == DataMode::TRAIN)
    {
        LOG(GREEN, "Feature column", (feat_cols.size() > 0 ? "s: " : ": "), YELLOW, feat_col_names);
        LOG(GREEN, "Target column", (tgt_cols.size() > 0 ? "s: " : ": "), YELLOW, tgt_col_names);
    }

    for (uint32 row = 0; row < m_csv->num_rows(); ++row)
    {
        std::vector<std::string> feature_strs;
        std::vector<std::string> target_strs;

        for (uint32 col : feat_cols)
        {
            feature_strs.push_back((*m_csv)(col, row));
        }
        for (uint32 col : tgt_cols)
        {
            target_strs.push_back((*m_csv)(col, row));
        }

        m_features.push_back(feature_parser(feature_strs, row));
        m_labels.push_back(target_parser(target_strs, row));
        m_indices.push_back(row);
    }

    this->set_num_batches(m_features.size() / m_features_node->batch());

    m_indices.resize(m_features_node->batch() * m_num_batches);
    std::shuffle(m_indices.begin(), m_indices.end(), rdm::det_gen);  // deterministic shuffle
    for (uint32 col : m_tgt_cols)
    {
        auto col_name = m_csv->get_column_name(col);
        m_csv->print_stats(col_name);
    }

    LOG(CYAN, m_indices.size(), " samples/", m_num_batches, " batches", RESET, " available in ",
        DataModeStr[uint32(mode)], (m_do_normalize ? GRAY " (normalized)" : ""));
}

#endif /* CSV_DATASET_HPP */
