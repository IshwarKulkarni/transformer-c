/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef EMOTION_DATA_HPP
#define EMOTION_DATA_HPP

#include "csv_dataset.hpp"
#include "dataset.hpp"
#include "nodes/unparameterized.hpp"
#include "string_utils.hpp"
#include "word2vec.hpp"

struct Words2VecParser
{
    const Word2VecBase& word2vec;
    const uint32 seq_len;
    std::map<uint32, uint32> m_lengths;  // length of each sentence, same order as CSV
    std::map<uint32, std::vector<std::string>> m_sentences;

    Words2VecParser(const Word2VecBase& word2vec, uint32 seq_len)
        : word2vec(word2vec), seq_len(seq_len)
    {
    }

    // split the line by space, and insert them to `words` upto max of `max_len`, return if there
    // were more words than `max_len`
    std::vector<std::string> split_line_to_words(std::string line)
    {
        std::vector<std::string> words;
        std::istringstream iss(line);
        std::string word;
        words.clear();
        uint32 word_count = 0;
        while (iss >> word)
        {
            if (word_count < seq_len) words.push_back(word);
            word_count++;
        }
        return words;
    }

    std::vector<FloatT> operator()(const std::vector<std::string>& sentences, uint32 row)
    {
        std::vector<FloatT> embeddings;
        m_sentences[row] = sentences;
        for (const auto& sentence : sentences)
        {
            auto words = split_line_to_words(sentence);
            m_lengths[row] = words.size();
            for (const auto& word : words)
            {
                auto node = word2vec[word];
                auto& src = node ? node->vec : WORDVEC{1};
                std::copy(src.begin(), src.end(), std::back_inserter(embeddings));
            }
        }
        return embeddings;
    }

    Optional<uint32> get_length(uint32 row) const
    {
        return m_lengths.find(row) != m_lengths.end() ? Optional<uint32>(m_lengths.at(row))
                                                      : Optional<uint32>();
    }
    Optional<std::vector<std::string>> get_sentences(uint32 row) const
    {
        return m_sentences.find(row) != m_sentences.end()
                   ? Optional<std::vector<std::string>>(m_sentences.at(row))
                   : Optional<std::vector<std::string>>();
    }
};

// Class that inherits from CSVDataset to load the text classification dataset
// Cannot have args in constructor to have "pre-parse" as true, as parsing cannot be done in the
// base class, CSVDataset.
struct TextClassification : public CSVDataset
{
    const Word2VecBase& m_word2vec;
    const uint32 m_seq_len;
    Words2VecParser m_word2vec_parser;

    //clang-format off
    TextClassification(std::string text_csv, const Word2VecBase* word2vec, const VarArgs& args,
                       Input<FloatT>* input_node = nullptr, Input<FloatT>* target_node = nullptr,
                       DataMode mode = DataMode::TRAIN)
        : CSVDataset(text_csv, mode, input_node, target_node, args.get("has_header", true),
                     args.get("delimiter", ','), args.get("quote", '"'),
                     args.get("max_samples", UINT32_MAX),
                     args.get("one_hot_classes", target_node->width()),
                     args.get("feature_columns", ""), args.get("target_columns", "")),
          m_word2vec(*word2vec),
          m_seq_len(input_node->height()),
          m_word2vec_parser(*word2vec, m_seq_len)
    //clang-format on
    {
        if (input_node->width() != word2vec->vector_dim())
        {
            throw_rte_with_backtrace("Data width must be ", word2vec->vector_dim(), ", but was ",
                                     input_node->width());
        }

        if (args.get("pre-parse", false) || args.get("normalize", false))
        {
            throw_rte_with_backtrace(
                "Pre-parse and normalize are not supported for text classification dataset");
        }
        Timer timer("Parsing Text to Features");
        StrsToOneHot target_parser{m_one_hot_classes};
        this->parse(m_word2vec_parser, target_parser);
    }

    void load(uint32 idx) override
    {
        m_features_node->set_val(std::numeric_limits<FloatT>::quiet_NaN());
        m_target_node->set_val(std::numeric_limits<FloatT>::quiet_NaN());
        CSVDataset::load(idx);
        for (uint32 i = 0; i < m_features_node->batch(); i++)
        {
            uint32 ext = m_word2vec_parser.m_lengths[m_last_load_indices[i]];
            if (ext > 0)
            {
                m_features_node->set_extent<HEIGHT_IDX>(i,
                                                        ext);  // set height to length of sentence
            }
            else
                throw_rte_with_backtrace("Sentence length is 0");
        }
    }

    void print_last_batch() const override
    {
        for (uint32 i = 0; i < m_features_node->batch(); i++)
        {
            uint32 csv_idx = m_last_load_indices[i];
            auto length = m_word2vec_parser.get_length(csv_idx);
            auto sentences = m_word2vec_parser.get_sentences(csv_idx);
            if (length && sentences)
            {
                std::string sentence_str = join(sentences.get(), " ");
                LOG(RED, i, " : ", csv_idx, " Length: ", length.get(), " Sentence: ", sentence_str);
            }
            else
            {
                throw_rte_with_backtrace("m_last_load_indices has invalid index: ", csv_idx);
            }
        }
    }
};

#endif  // EMOTION_DATA_HPP
