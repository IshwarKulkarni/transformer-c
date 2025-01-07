#include <iostream>
#include "matrix.cuh"

int train_MLP()  // train to generate a zero vector
{
    Input<> inp1(5, 10, "inp1");
    FeedForward<> Model({5, &inp1, true, "Lin1"}, {3, nullptr, false, "Lin2"}, 0.25, "MLP1");
    Input<> target(Model.shape(), "target");
    L2Loss<> loss({&Model, &target}, "L2Loss");
    fillCPU(target, 0);
    xavier_uniform_init<FloatT>(inp1);
    std::ofstream dot("simple.dot");
    graph_to_dot(&loss, dot);

    uint32 max_iter = 2;
    for (uint32 i = 0; i < max_iter; i++)
    {
        LOG(GREEN, "Iteration: ", i);
        xavier_uniform_init<FloatT>(inp1);
        loss.compute();
        loss.backward();
        if ((i + 1) % 10 == 0)
        {
            std::cout << i << ", " << loss.value() << std::endl;
        }
        LOG_SYNC("W1_grads", Model.l_in->W.grads());
        LOG_SYNC("W1", Model.l_in->W);
        loss.update_weights(0.01);
        LOG_SYNC("W1", Model.l_in->W);
    }
    return 0;
}

int main()
{
    Matrix<float> A({2, 2}, "A");
    std::cout << A;
    return 0;
}