/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "csv_dataset.hpp"
#include "dataset.hpp"
#include "errors.hpp"
#include "logger.hpp"
#include "types"
#include "utils.hpp"

const char* DataModeStr[] = {"Train", "Validation", "Test"};

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

    uint32 malformed_lines = 0;
    while (data.size() < max_samples && file.good())
    {
        std::vector<std::string> tokens;
        for (uint32 i = 0; i < column_count && file.good(); ++i)
        {
            char delim = i == column_count - 1 ? '\n' : delimiter;
            if (std::getline(file, line, delim)) tokens.push_back(line);
        }
        if (tokens.size() != column_count)
        {
            malformed_lines++;
        }
        else
            data.push_back(tokens);
    }
    std::string malformed =
        malformed_lines > 0 ? " with " + std::to_string(malformed_lines) + " malformed lines" : "";
    LOG("Read ", GREEN, commas_int(data.size()), RESET, " rows from ", GREEN, filename, RESET,
        " with ", GREEN, column_count, RESET, " columns", RED, malformed);
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
    std::stringstream ss;
    for (const auto& [value, count] : unique_counts)
    {
        ss << value << " : " << count << "\t";
    }
    LOG(GREEN, ss.str());
}

// print histogram of the column, using "*"s to represent the count , or print counts.
void CSV::print_stats(const std::string& col, Optional<uint32> num_bins)
{
    if (header_to_index.find(col) == header_to_index.end())
    {
        throw_rte_with_backtrace("Column ", col, " not found in the csv file");
    }

    uint32 col_idx = get_column_index(col);
    std::map<FloatT, uint32> counts;

    std::vector<FloatT> values;
    FloatT min = std::numeric_limits<FloatT>::max(), max = std::numeric_limits<FloatT>::min();
    for (uint32 row = 0; row < num_rows(); ++row)
    {
        std::string str_val = data[row][col_idx];
        FloatT val = string_to_type<FloatT>(str_val);
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
            // TODO: this cast to uint32 is wrong when T is not an integer type
            counts_str[std::to_string(uint32(value))] = count;
        }
        std::stringstream ss;
        for (const auto& [value, count] : counts_str)
        {
            ss << value << ": " << uint32(count) << "|\t";
        }
        LOG(ss.str());
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
//clang-format off
CSVDataset::CSVDataset(std::string csv_filename, DataMode mode, Input<FloatT>* features_node,
                       Input<FloatT>* target_node, bool has_header, char delimiter, char quote,
                       uint32 max_samples, uint32 one_hot_classes, const std::string& feature_col,
                       const std::string& target_col)
    : Dataset(mode, mode == DataMode::TRAIN && false, features_node, target_node),
      m_features_node(features_node),
      m_target_node(target_node),
      m_one_hot_classes(0)
{
    //clang-format on
    m_csv = std::make_unique<CSV>(csv_filename, has_header, delimiter, quote, max_samples);

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

    m_tgt_cols = cols_to_indices(feature_col);
    m_feat_cols = cols_to_indices(target_col);

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

    if (one_hot_classes > 0)
    {
        // TODO: return std::vector<std::vector<FloatT>> for multiple target columns
        if (m_tgt_cols.size() != 1)
        {
            throw_rte_with_backtrace(
                "One-hot encoding is only supported for a single target column");
        }
        m_one_hot_classes = one_hot_classes;
    }
}

CSVDataset::CSVDataset(std::string csv_filename, DataMode mode, Input<FloatT>* features_node,
                       Input<FloatT>* target_node, const VarArgs& args)
    : CSVDataset(csv_filename, mode, features_node, target_node, args.get("has_header", true),
                 args.get("delimiter", ','), args.get("quote", '"'),
                 args.get("max_samples", UINT32_MAX), args.get("one_hot_classes", 0),
                 args.get("feature_columns", ""), args.get("target_columns", ""))
{
    if (m_one_hot_classes != 0 && m_one_hot_classes != m_target_node->width())
    {
        throw_rte_with_backtrace("Target node width must be ", m_one_hot_classes,
                                 " for one-hot encoding target of width ", m_target_node->width());
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
        StrsToFloats str_to_float;
        if (m_one_hot_classes > 0)
        {
            StrsToOneHot target_parser{m_one_hot_classes};
            parse(str_to_float, target_parser);
        }
        else
        {
            parse(str_to_float, str_to_float);
        }
    }
    if (m_do_normalize)
    {
        normalize();
    }
}

// normalize features in place
void CSVDataset::normalize()
{
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
    uint32 batch_size = m_features_node->batch();
    populate_next_batch(batch_idx, batch_size);

    for (uint32 i = 0; i < batch_size; ++i)
    {
        auto data_idx = m_last_load_indices[i];
        m_features[data_idx].resize(m_features_node->shape.size2d,
                                    std::numeric_limits<FloatT>::quiet_NaN());
        m_labels[data_idx].resize(m_target_node->shape.size2d,
                                  std::numeric_limits<FloatT>::quiet_NaN());
        m_features_node->copy(m_features[data_idx].data(), i);
        m_target_node->copy(m_labels[data_idx].data(), i);
    }
}
