#ifndef WORD2VEC_HPP
#define WORD2VEC_HPP

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include "types"

static constexpr uint32 N = 300;

static constexpr FloatT L2D_DIST_THRESHOLD = FloatT(1e-2);
// This is not right:
static constexpr FloatT COSINE_SIM_THRESHOLD = FloatT(1) - 1e-4;

typedef std::array<float32, N> Vec300;
typedef std::pair<std::string, Vec300> WordVecPair;
typedef std::vector<WordVecPair>::iterator WordVecPairIter;

std::ostream& operator<<(std::ostream& os, const Vec300& arr);
std::istream& operator>>(std::istream& is, Vec300& arr);

static constexpr std::array<float32, 100> principal_comps = {
    167, 81,  243, 66,  77,  133, 281, 87,  295, 266, 190, 191, 224, 191, 74,  61,  156,
    265, 178, 186, 203, 277, 1,   274, 115, 168, 240, 245, 136, 102, 114, 149, 127, 282,
    249, 105, 151, 278, 132, 217, 238, 187, 263, 239, 22,  180, 236, 120, 151, 271, 211,
    216, 255, 207, 150, 63,  94,  207, 181, 99,  241, 260, 248, 253, 292, 23,  134, 155,
    132, 78,  39,  43,  30,  218, 298, 98,  202, 25,  69,  155, 153, 235, 65,  199, 107,
    244, 256, 7,   132, 58,  207, 166, 14,  239, 244, 98,  60,  6,   35,  236};

static constexpr uint32 N_PCA = 30;

static_assert(N_PCA <= principal_comps.size(), "N_PCA is too large");

static constexpr uint32 get_axis(uint32 depth) { return principal_comps[depth % N_PCA]; }

inline FloatT l2_dist2_pca(const Vec300& a, const Vec300& b, uint32 L = N_PCA)
{
    FloatT dist = 0;
    for (uint32 i = 0; i < L; i++)
    {
        dist += (a[principal_comps[i]] - b[principal_comps[i]]) *
                (a[principal_comps[i]] - b[principal_comps[i]]);
    }
    return dist;
}

inline FloatT l2_dist2(const Vec300& a, const Vec300& b)
{
    FloatT dist = 0;
    for (uint32 i = 0; i < N; i++)
    {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return dist;
}

template <uint32 LD>
inline std::array<float32, LD> vec300ToVecLD(const Vec300& vec)
{
    std::array<float32, LD> vecLD;
    static constexpr uint32 chunk_size = N / LD;
    for (uint32 i = 0; i < LD; i++)
    {
        vecLD[i] =
            std::accumulate(vec.begin() + i * chunk_size, vec.begin() + (i + 1) * chunk_size, 0.0);
    }
    return vecLD;
}

std::array<std::pair<float32, bool>, N> diff(const Vec300& a, const Vec300& b, float32 eps = 1e-6,
                                             uint32* count = nullptr);

inline FloatT cos_sim(const Vec300& a, const Vec300& b)
{
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

inline Vec300 make_vec300(const FloatT* arr)
{
    Vec300 vec;
    std::copy(arr, arr + N, vec.begin());
    return vec;
}

FloatT add_noise(Vec300& vec, FloatT mean = 0, FloatT std = 1, uint32* seed = nullptr);

FloatT add_noise_normal(Vec300& vec, FloatT mean = 0, FloatT std = 1, uint32* seed = nullptr);

struct WordVecNode
{
    const std::string word;
    const Vec300 vec;
    const uint16 depth;
    const uint32 id = get_id();
    WordVecNode *left = nullptr, *right = nullptr, *parent = nullptr;
    WordVecNode(const WordVecPair& tuple, uint16 depth)
        : word(std::get<0>(tuple)), vec(std::get<1>(tuple)), depth(depth)
    {
    }
    FloatT dist2(const Vec300& other) const { return l2_dist2(vec, other); }
    FloatT dist2(const WordVecNode* node) const { return l2_dist2(vec, node->vec); }
    FloatT cos_sim(const Vec300& other) const { return ::cos_sim(vec, other); }
    FloatT cos_sim(const WordVecNode* node) const { return ::cos_sim(vec, node->vec); }

 private:
    static uint32 get_id()
    {
        static uint32 id = 0;
        return id++;
    }
};
typedef std::vector<const WordVecNode*> NodeVector;
typedef std::pair<const WordVecNode*, FloatT> NodeDist2;  // node and similarity

void dfs(const WordVecNode* start_node, NodeVector& out);

void print_path(const WordVecNode* node);

inline bool is_left_child(const WordVecNode* node)
{
    return node and node->parent and node->parent->left == node;
}

std::vector<const WordVecNode*> get_ancestors(const WordVecNode* node, bool root_first = true);

inline bool operator==(const WordVecNode& a, const WordVecNode& b) { return a.id == b.id; }

std::ostream& operator<<(std::ostream& os, const WordVecNode& node);

enum SearchOption  // for Word2Vec::nearest, tested by adding sigma=0.01 noise to the 200K vectors
{
    FAST = 0,
    ACCURATE = 1,
    EXACT = 2
};
struct Word2Vec
{
    Word2Vec(const std::string& filename);

    size_t size() const { return word2Node.size(); }

    void nearest(const Vec300& vec, const WordVecNode* from, NodeDist2& out,
                 FloatT thresh = COSINE_SIM_THRESHOLD, uint32* count = nullptr,
                 uint32 count_threshold = 1000, uint32 maxDepthForExact = 8);

    const WordVecNode* operator[](const Vec300& vec);

    inline const WordVecNode* operator[](uint32 id) const
    {
        return id < this->nodes.size() ? nodes[id] : nullptr;
    }

    inline const WordVecNode* operator[](const std::string& word) const
    {
        auto it = word2Node.find(word);
        return it != word2Node.end() ? it->second : nullptr;
    }

    const WordVecNode* operator()(const Vec300& vec, SearchOption option = SearchOption::ACCURATE);

    uint32 nearest_count = 0;

    ~Word2Vec()
    {
        for (WordVecNode* node : nodes) delete node;
    }

 private:
    WordVecNode* build(WordVecPairIter begin, WordVecPairIter end, int depth);
    const WordVecNode* root;
    std::vector<WordVecNode*> nodes;
    std::unordered_map<std::string, const WordVecNode*> word2Node;
};

#endif  // WORD2VEC_HPP
