#ifndef IRISI_DATA_HPP
#define IRISI_DATA_HPP

#include <string>
#include <vector>
#include "nodes.hpp"

struct CSVDataset
{
    using Feature = std::array<float32, 4>;
    using Label = std::array<float32, 3>;
    std::vector<Feature> features;
    std::vector<Label> labels;
    std::vector<uint32> all_indices;
    uint32 loaded_batches = 0;

    Input<> sample;
    Input<> target;

    const std::string label_names[3] = {"Setosa", "Versicolor", "Virginica"};

    CSVDataset(std::string iris_csv, uint32 batch, uint32 max_samples = -1)
        : sample({batch, 1, 4, "sample"}), target({batch, 1, 3, "target"})
    {
        std::ifstream iris_file(iris_csv);
        if (!iris_file)
        {
            throw_rte_with_backtrace("Could not open file: ", iris_csv);
        }
        std::string header;
        std::getline(iris_file, header);
        while (iris_file && features.size() < max_samples)
        {
            Feature feature;
            char comma;
            for (uint32 i = 0; i < 4; i++)
            {
                iris_file >> feature[i] >> comma;
            }
            std::string label;
            iris_file >> label;
            if (comma != ',' or label == "") break;
            features.push_back(feature);
            labels.push_back({float32(label == label_names[0]), float32(label == label_names[1]),
                              float32(label == label_names[2])});
        }
        LOG(GREEN, "Read ", features.size(), " from ", iris_csv);
        all_indices.resize(features.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        swizzle();
    }

    void swizzle() { std::shuffle(std::begin(all_indices), std::end(all_indices), rdm::gen()); }

    uint32 size() { return features.size(); }

    void load(uint32 idx)
    {
        loaded_batches++;
        if (loaded_batches % size() == 0)
        {
            swizzle();
        }
        idx = all_indices[idx % all_indices.size()];
        std::copy(features[idx].begin(), features[idx].end(), sample.begin());
        std::copy(labels[idx].begin(), labels[idx].end(), target.begin());
    }
};

#endif
