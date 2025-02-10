#ifndef DATA_NODES
#define DATA_NODES

#include <set>
#include <vector>
#include "nodes/unparameterized.hpp"

// train or test data enum:
enum class DataMode
{
    TRAIN = 0,
    TEST = 1
};

static const char* DataModeStr[] = {"Train", "Test"};

// Read a CSV file where first line is header and the rest are data
// The last column is the label(any string), and the rest are features
struct CSVDataset
{
 protected:
    std::vector<std::vector<FloatT>> features;
    std::vector<std::vector<FloatT>> labels;
    std::vector<std::string> label_strings;

    std::vector<uint32> train_indices, valdn_indices;

    uint32 train_batches = 0;
    uint32 valdn_batches = 0;
    uint32 batch = 0;

    uint32 num_features = 0;

    std::unique_ptr<Input<>> input_node;
    std::unique_ptr<Input<>> target_node;

    virtual void make_nodes() = 0;

 public:
    CSVDataset(std::string csv_filename, uint32 batch, float32 validation_ratio = 0.1f,
               uint32 max_samples = UINT32_MAX)
        : batch(batch)
    {
        std::ifstream csv_file(csv_filename);
        if (!csv_file)
        {
            throw_rte_with_backtrace("Could not open file: ", csv_filename);
        }
        std::string header;
        std::getline(csv_file, header);

        num_features = std::count(header.begin(), header.end(), ',');
        // num features is one more than the number of commas, but last column is label

        if (num_features == 0) throw_rte_with_backtrace("No features in ", csv_filename);

        while (csv_file && features.size() < max_samples)
        {
            std::vector<FloatT> feature(num_features);
            char comma;
            for (uint32 i = 0; i < num_features; i++) csv_file >> feature[i] >> comma;
            std::string label;
            csv_file >> label;
            if (comma != ',' or label == "") break;
            features.push_back(feature);
            label_strings.push_back(label);
        }

        if (features.empty()) throw_rte_with_backtrace("No Features loaded");

        std::vector<uint32> all_indices;
        all_indices.resize(features.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::random_shuffle(all_indices.begin(), all_indices.end());

        uint32 valdn_size = features.size() * validation_ratio;
        valdn_indices.resize(valdn_size);
        std::copy(all_indices.begin(), all_indices.begin() + valdn_size, valdn_indices.begin());

        train_indices.resize(features.size() - valdn_size);
        std::copy(all_indices.begin() + valdn_size, all_indices.end(), train_indices.begin());

        train_batches = train_indices.size() / batch;
        valdn_batches = valdn_indices.size() / batch;

        if (train_batches == 0 or valdn_batches == 0)
            throw_rte_with_backtrace("Not enough samples for training(", train_batches,
                                     ") or validation(", valdn_batches, ")");

        swizzle(DataMode::TRAIN);
        swizzle(DataMode::TEST);
    }

    void normalize()
    {
        std::vector<FloatT> column;
        column.reserve(features.size());
        for (uint32 i = 0; i < num_features; i++)
        {
            column.clear();
            for (auto& feature : features)
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
            for (auto& feature : features)
            {
                feature[i] = (feature[i] - mean) / std_dev;
            }
        }
    }

    void swizzle(DataMode mode)
    {
        auto& indices = mode == DataMode::TRAIN ? train_indices : valdn_indices;
        std::random_shuffle(indices.begin(), indices.end());
    }

    uint32 num_batches(DataMode mode) const
    {
        return mode == DataMode::TRAIN ? train_batches : valdn_batches;
    }

    // Load the data into the input and target nodes, with the given index (modulo the number of
    // batches in the mode).
    std::pair<const Input<>&, const Input<>&> load(DataMode mode, uint32 idx)
    {
        if (!input_node or !target_node) throw_rte_with_backtrace("Nodes not initialized");

        auto& indices = mode == DataMode::TRAIN ? train_indices : valdn_indices;

        if (idx % indices.size() == 1) swizzle(mode);

        if (idx >= indices.size())
            throw_rte_with_backtrace("Index out of bounds: ", idx, " >= ", indices.size(),
                                     DataModeStr[static_cast<uint32>(mode)], " mode ");

        for (uint32 i = 0; i < input_node->batch(); i++)
        {
            input_node->copy(features[indices[idx + i]].data(), i);
            target_node->copy(labels[indices[idx + i]].data(), i);
        }
        return {*input_node.get(), *target_node.get()};
    }

    inline Input<>* input() { return input_node.get(); }
    inline Input<>* target() { return target_node.get(); }

    uint32 input_size() const { return num_features; }
    uint32 target_size() const { return target_node->width(); }
};

// The last column is the category label(any string), and the rest are features
struct CategoricalDataset : public CSVDataset
{
 private:
    std::set<std::string> uniq_labels;
    uint32 num_classes = 0;

 public:
    CategoricalDataset(std::string csv_filename, uint32 batch, float32 validation_ratio = 0.1f,
                       uint32 max_samples = UINT32_MAX)
        : CSVDataset(csv_filename, batch, validation_ratio, max_samples)
    {
        uniq_labels.insert(label_strings.begin(), label_strings.end());
        std::vector<uint32> uniq_label_ct(uniq_labels.size(), 0);

        num_classes = uniq_labels.size();

        for (auto& label : label_strings)
        {
            std::vector one_hot(num_classes, 0.f);
            size_t idx = std::distance(uniq_labels.begin(), uniq_labels.find(label));
            one_hot[idx] = 1;
            labels.push_back(one_hot);
            uniq_label_ct[idx]++;
        }

        std::string label_counts;
        auto label_it = uniq_labels.begin();
        for (uint32 i = 0; i < num_classes; i++)
        {
            std::string ct = std::to_string(uniq_label_ct[i]);
            label_counts += *(label_it++) + ":" + ct + " samples\t";
        }

        LOG(GREEN, "Read ", features.size(), " samples with ", num_features, " features and ",
            num_classes, " classes", " split into ", train_batches, " train and ", valdn_batches,
            " validation batches", " from ", csv_filename);
        LOG(YELLOW, "Labels counts:\t", label_counts);

        make_nodes();
    }

    void make_nodes() override
    {
        input_node = std::make_unique<Input<>>(this->batch, 1u, num_features, "input");
        target_node = std::make_unique<Input<>>(this->batch, 1u, num_classes, "target");
    }
};

// Last column is continuos scalar, approporiate for regression tasks
struct ContinuousDataset : public CSVDataset
{
    ContinuousDataset(std::string csv_filename, uint32 batch, float32 validation_ratio = 0.1f,
                      uint32 max_samples = UINT32_MAX)
        : CSVDataset(csv_filename, batch, validation_ratio, max_samples)
    {
        for (auto& label : label_strings)
        {
            FloatT val = std::stof(label);
            labels.push_back({val});
        }

        auto [min_label, max_label] =
            std::minmax_element(labels.begin(), labels.end(),
                                [&](const auto& a, const auto& b) { return a[0] < b[0]; });
        auto min = min_label->begin(), max = max_label->begin();

        LOG(GREEN, "Read ", features.size(), " samples with ", num_features, " features and ",
            " split into ", train_batches, " train and ", valdn_batches, " validation batches",
            " from ", csv_filename);
        LOG(YELLOW, "Label range:\t", *min, " to ", *max);

        make_nodes();
    }

    void make_nodes() override
    {
        input_node = std::make_unique<Input<>>(this->batch, 1u, num_features, "input");
        target_node = std::make_unique<Input<>>(this->batch, 1u, 1u, "target");
    }
};

#endif
