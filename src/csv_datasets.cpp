/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "csv_dataset.hpp"
#include "dataset.hpp"
#include "errors.hpp"
#include "logger.hpp"
#include "types"
#include "utils.hpp"

template <typename T, typename U>
inline std::string join(const std::map<T, U>& strs, const std::string& delim = ", ")
{
    std::string result;
    for (const auto& [key, value] : strs)
    {
        result += key + " : " + MAGENTA + std::to_string(value) + RESET + delim;
    }
    return result;
}

CSV::CSV(std::string filename, bool has_header, char delimiter, char quote, uint32 max_samples)
    : delimiter(delimiter), quote(quote), has_header(has_header)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw_rte_with_backtrace("Failed to open file: ", filename);
    }
    std::string line;
    std::getline(file, line);
    header = split_str(line, delimiter, quote);
    for (auto& col : header) col = strip_whitespace(col);
    uint32 column_count = header.size();

    if (has_header)
    {
        for (uint32 i = 0; i < column_count; ++i)
        {
            header_to_index[header[i]] = i;
        }
    }
    else
    {
        for (uint32 i = 0; i < column_count; ++i)
        {
            auto str_i = std::to_string(i);
            header_to_index[str_i] = i;
            header[i] = str_i;
        }
        data.push_back(header);
    }

    while (std::getline(file, line) && data.size() < max_samples)
    {
        auto tokens = split_str(line, delimiter, quote);
        if (tokens.size() != column_count)
        {
            LOG(RED, "Number of columns in the csv file: ", tokens.size());
            throw std::runtime_error("Number of columns in the csv file is not consistent");
        }
        data.push_back(tokens);
    }
    LOG("Read ", GREEN, data.size(), RESET, " rows from ", GREEN, filename, RESET, " with ", GREEN,
        column_count, RESET, " columns");
}

void CSV::print_counts(const std::string& col)
{
    if (header_to_index.find(col) == header_to_index.end())
    {
        throw_rte_with_backtrace("Column ", col, " not found in the csv file");
    }

    uint32 col_idx = get_column_index(col);

    std::map<std::string, uint32> unique_counts;
    for (uint32 row = 0; row < num_rows(); ++row)
    {
        unique_counts[data[row][col_idx]]++;
    }
    LOG(GREEN, join(unique_counts, "\t"));
}

// print histogram of the column, using "*"s to represent the count ,
template <typename T>
void CSV::print_stats(const std::string& col, Optional<uint32> num_bins)
{
    if (header_to_index.find(col) == header_to_index.end())
    {
        throw_rte_with_backtrace("Column ", col, " not found in the csv file");
    }

    uint32 col_idx = get_column_index(col);
    std::map<T, uint32> counts;

    std::vector<T> values;
    FloatT min = std::numeric_limits<FloatT>::max(), max = std::numeric_limits<FloatT>::min();
    for (uint32 row = 0; row < num_rows(); ++row)
    {
        std::string str_val = data[row][col_idx];
        T val = string_to_type<T>(str_val);
        counts[val]++;
        values.push_back(val);
        min = std::min(min, string_to_type<FloatT>(str_val));
        max = std::max(max, string_to_type<FloatT>(data[row][col_idx]));
    }

    if (counts.size() < 10)
    {
        std::map<std::string, uint32> counts_str;
        for (const auto& [value, count] : counts)
        {
            counts_str[std::to_string(value)] = count;
        }
        LOG(GREEN, join(counts_str, "\t"));
        return;
    }

    uint32 max_width = 80;  // at max print 80 *s
    uint32 num_bins_val = num_bins.value_or(10);
    std::vector<uint32> bin_counts(num_bins_val, 0);
    for (const auto& [value, count] : counts)
    {
        uint32 bin_idx = (value - min) / (max - min) * num_bins_val;
        bin_counts[bin_idx] += count;
    }

    std::stringstream ss;
    char line[160];

    uint32 max_count = *std::max_element(bin_counts.begin(), bin_counts.end());

    for (uint32 i = 0; i < bin_counts.size(); ++i)
    {
        uint32 count = bin_counts[i];
        uint32 graph_len = count * max_width / max_count;

        graph_len += (count > 0 && graph_len == 0) ? 1 : 0;

        FloatT bin_min = min + i * (max - min) / num_bins_val;
        FloatT bin_max = min + (i + 1) * (max - min) / num_bins_val;

        snprintf(line, 120, "%2d|  %5.2f - %5.2f: %6d : %s\n", i, bin_min, bin_max, count,
                 std::string(graph_len, '>').c_str());
        ss << line;
    }
    LOG(GREEN, "\n", "stats for column `", col, "`\n", ss.str());
}

CSVDataset::CSVDataset(std::string csv_filename, DataMode mode, Input<FloatT>* features_node,
                       Input<FloatT>* target_node, const VarArgs& args)
    : Dataset(mode, mode == DataMode::TRAIN, features_node, target_node),
      m_features_node(features_node),
      m_target_node(target_node),
      m_one_hot_classes(0)
{
    char delimiter = args.get("delimiter", ',');
    char quote = args.get("quote", '"');

    m_csv = std::make_unique<CSV>(csv_filename, args.get("has_header", true), delimiter, quote,
                                  args.get("max_samples", UINT32_MAX));

    if (m_csv->num_columns() == 0 || m_csv->num_rows() == 0)
    {
        throw_rte_with_backtrace("CSV file ", csv_filename, " is empty");
    }

    auto cols_to_indices = [&](std::string col_arg) {
        auto col_names = split_str(col_arg, delimiter, quote);
        std::set<uint32> col_indices;
        for (const auto& col : col_names)
        {
            col_indices.insert(m_csv->get_column_index(col));
        }
        return col_indices;
    };

    m_tgt_cols = cols_to_indices(args.get("target_columns", ""));
    m_feat_cols = cols_to_indices(args.get("feature_columns", ""));

    if (m_tgt_cols.empty())
    {
        m_tgt_cols = {m_csv->num_columns() - 1};
    }

    if (m_feat_cols.empty())
    {
        for (uint32 col = 0; col < m_csv->num_columns(); ++col)
        {
            if (m_tgt_cols.find(col) == m_tgt_cols.end())
            {
                m_feat_cols.insert(col);
            }
        }
    }

    for (uint32 col = 0; col < m_csv->num_columns(); ++col)
    {
        if (m_tgt_cols.find(col) == m_tgt_cols.end() && m_feat_cols.find(col) == m_feat_cols.end())
        {
            LOG(YELLOW, "Column ", m_csv->get_column_name(col), " is not used");
        }
    }

    if (auto one_host_classes = args.get("one_hot_classes", 0))
    {
        // TODO: return std::vector<std::vector<FloatT>> for multiple target columns
        if (m_tgt_cols.size() != 1)
        {
            throw_rte_with_backtrace(
                "One-hot encoding is only supported for a single target column");
        }
        m_one_hot_classes = one_host_classes;
    }

    if (m_one_hot_classes != 0 && m_one_hot_classes != m_target_node->width())
    {
        throw_rte_with_backtrace("Target node width must be ", m_one_hot_classes,
                                 " for one-hot encoding");
    }
    else if (m_one_hot_classes == 0 && m_tgt_cols.size() != m_target_node->width())
    {
        throw_rte_with_backtrace(
            "Target node width must be equal to the number of target columns: ", m_tgt_cols.size(),
            " != ", m_target_node->width());
    }

    if (m_feat_cols.size() != m_features_node->width())
    {
        throw_rte_with_backtrace(
            "Feature node width must be equal to the number of feature columns: ",
            m_feat_cols.size(), " != ", m_features_node->width());
    }

    m_do_normalize = args.get("normalize", false);
    if (args.get("pre-parse", true))
    {
        if (m_one_hot_classes > 0)
        {
            parse(StrsToFloats(), StrsToOneHot{m_one_hot_classes});
        }
        else
        {
            parse(StrsToFloats(), StrsToFloats());
        }
    }
}

template <typename FeatureParser, typename TargetParser>
void CSVDataset::parse(const FeatureParser& feature_parser, const TargetParser& target_parser)
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
            feat_col_names += m_csv->get_column_name(col) + ", ";
        }
        if (m_tgt_cols.find(col) != m_tgt_cols.end())
        {
            tgt_cols.push_back(col);
            tgt_col_names += m_csv->get_column_name(col) + ", ";
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

        m_features.push_back(feature_parser(feature_strs));
        m_labels.push_back(target_parser(target_strs));
    }
    if (m_do_normalize)
    {
        normalize();
    }

    m_indices.resize(m_features.size());
    std::iota(m_indices.begin(), m_indices.end(), 0);

    for (uint32 col : m_tgt_cols)
    {
        auto col_name = m_csv->get_column_name(col);
        m_csv->print_stats<FloatT>(col_name);
    }

    this->set_num_batches(m_features.size() / m_features_node->batch());
}

// normalize features in place
void CSVDataset::normalize()
{
    LOG(GREEN, "Normalizing features");
    std::vector<FloatT> column;
    column.reserve(m_features.size());
    for (uint32 i = 0; i < m_features[0].size(); i++)
    {
        column.clear();
        for (auto& feature : m_features)
        {
            column.push_back(feature[i]);
        }
        FloatT mean = std::accumulate(column.begin(), column.end(), 0.f) / column.size();
        FloatT std_dev = 0;
        for (auto& val : column)
        {
            std_dev += (val - mean) * (val - mean);
        }
        std_dev = std::sqrt(std_dev / column.size());
        for (auto& feature : m_features)
        {
            feature[i] = (feature[i] - mean) / std_dev;
        }
    }
}

void CSVDataset::load(uint32 batch_idx)
{
    if (batch_idx % m_indices.size() == 0 && m_shuffle)
    {
        std::shuffle(m_indices.begin(), m_indices.end(), rdm::rdm_gen);
    }

    uint32 batch_size = m_features_node->batch();
    uint32 start_idx = batch_idx * batch_size;

    for (uint32 i = 0; i < batch_size; ++i)
    {
        auto data_idx = m_indices[(start_idx + i) % m_indices.size()];
        m_features_node->copy(m_features[data_idx].data(), i);
        m_target_node->copy(m_labels[data_idx].data(), i);
    }
}
