#ifndef NETWORK_TRAINER_HPP
#define NETWORK_TRAINER_HPP

#include "dataset.hpp"
#include "network_graph.hpp"

class NetworkTrainer
{
 public:
    NetworkTrainer(NetworkGraph& graph) : m_graph(graph) {}
    void train() {}

    void validate() {}

    const NetworkGraph& graph() const { return m_graph; }

 private:
    NetworkGraph& m_graph;
};

#endif
