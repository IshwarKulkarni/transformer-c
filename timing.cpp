#include "nodes/loss.hpp"
#include "nodes/parameterized.hpp"

using MatrixT = Matrix<FloatT>;

using namespace std;

template <typename Func>
float64 flushing_exec(uint32 max_iters, std::string name, Func f)
{
    float64 time = 0;
    uint64 bytes_processed = 0;
    for (uint32 i = 0; i < max_iters; i++)
    {
        // clear_l2_cache();
        // CudaEventTimer timer("");
        bytes_processed += f();
        // time += timer.stop();
    }

    float64 bandWidth_mb = bytes_processed / (time * (1 << 30));
    LOG(name, " Bandwidth: ", BLUE, std::setprecision(5), std::setw(7), bandWidth_mb, " GB/s ",
        RESET, " Time: ", YELLOW, time * 1000 / max_iters, "ms. ", RESET, " per call.");
    return time;
}

void run_reduce_timing()
{
    // clang-format off
    std::vector<Shape> shapes = {Shape{31, 31, 16},
                                 Shape{129, 258, 258},
                                 Shape{33, 257, 514},
                                 Shape{65, 1025, 1024},
                                 Shape{65, 3300, 3000}};
    for (auto& s : shapes)
    {
        auto A = xavier_uniform_init<FloatT>(s);
        uint32 bytes = A.numels() * sizeof(FloatT);

        std::string n = "Reduction of "  RED + s.str() + RESET + " along dim ";
        MatrixT R0(A.shape.set(0, 1));
        MatrixT R1(A.shape.set(1, 1));
        MatrixT R2(A.shape.set(2, 1));

        flushing_exec(10, n + "0", [&]() {reduce<FloatT, 0> (A, A); return bytes;});
        flushing_exec(10, n + "1", [&]() {reduce<FloatT, 1> (A, A); return bytes;});
        flushing_exec(10, n + "2", [&]() {reduce<FloatT, 2> (A, A); return bytes;});
    }
    // clang-format on
}

void time_linear_node()
{
    uint32 batch = 16;
    uint32 Ei = 4 * 128;
    uint32 Sl = 4 * 64;
    uint32 I1 = 4 * 256;
    uint32 I2 = 4 * 128;
    Input<> x(batch, Sl, Ei, "x");

    Linear<FloatT> L0(LinearInput<FloatT>{I1, &x, true, "Sigmoid", "Linear-L0"});
    Linear<FloatT> L1(LinearInput<FloatT>{I2, &L0, true, "TanH", "Linear-L1"});
    SoftmaxDim0<FloatT> softmax(&L1, "S");
    Input<> t(L1.shape, "target");
    NLLLoss<> loss({&softmax, &t}, "L2Error");

    uint64 bytes = L0.shape.bytes<FloatT>() + L1.shape.bytes<FloatT>();

    uint32 max_iters = 3;
    uint32 iters = 0;
    float64 time = flushing_exec(max_iters, "Linear Node", [&]() {
        normal_init(x, 1, 2 * iters);
        loss.compute();
        loss.backward();
        return bytes;
    });

    LOG(GREEN, "Time per iteration: ", time * 1000.f / max_iters, "ms, total time: ", time, "s");
}

void run_transpose_timing(const MatrixT& A)
{
    uint32 bytes = A.numels() * sizeof(FloatT);
    std::string name = "Transpose of " RED + A.shape.str() + RESET;
    MatrixT At(A.shape.t());
    flushing_exec(10, name, [&]() {
        transpose<FloatT>(At, A);
        return bytes;
    });
}

void run_mm_timing(const MatrixT& A, const MatrixT& B)
{
    Matrix<FloatT> AB({A.batch(), A.height(), B.width()});
    uint32 bytes = AB.shape.bytes<FloatT>();

    std::string name =
        "MMADD of \t" RED + A.shape.str() + RESET + " and " + RED + B.shape.str() + RESET;
    flushing_exec(10, name, [&]() {
        mmadd<FloatT>(AB, A, B);
        return bytes;
    });

    Matrix<FloatT> Bt(B.shape.t());
    Matrix<FloatT> ABt({A.batch(), A.height(), Bt.height()});
    name = "MMTADD of \t" RED + A.shape.str() + RESET + " and " + RED + Bt.shape.str() + RESET;
    flushing_exec(10, name, [&]() {
        mmTadd<FloatT>(ABt, A, Bt);
        return bytes;
    });
}

void run_mm_timing()
{
    // clang-format off
    std::pair<Shape, uint32> shapes[] = {
        //{Shape(6, 12, 14), 10},
        //{Shape(8, 512, 256), 128},
        //{Shape(12, 512, 512), 384},
        {Shape(32, 2048, 512), 2048},
        {Shape(33, 2050, 515), 515},
        {Shape(16, 2048, 2048), 2048},
        {Shape(9, 2048, 1024), 1},
        {Shape(10, 1100, 1200), 900}, 
        {Shape(8, 4096, 4096), 1024},
    };
    // clang-format on

    for (auto shape : shapes)
    {
        auto s = shape.first;
        uint32 k = shape.second;
        auto A = xavier_uniform_init<FloatT>(s);
        auto B = xavier_uniform_init<FloatT>(A.shape.t());
        run_mm_timing(A, B);

        auto B2 = xavier_uniform_init<FloatT>({A.batch(), A.width(), k});
        run_mm_timing(A, B2);
    }
}

int time_attention()
{
    uint32 bn = 16;    //  batch size
    uint32 Ei = 1024;  //  input embedding size
    uint32 Eq = 256;   //  query embedding size
    uint32 Ev = Ei;    //  value, i.e. output embedding size for each head
    uint32 Sl = 20;    //  sequence length

    // clang-format off
     Input<> q(bn, Sl, Ei, "Qi"), 
             k(bn, Sl, Ei, "Ki"),
             v(bn, Sl, Ei, "Vi");

    Attention<> A({Eq, &q, false, "identity", "Attention_Q"}, 
                  {Eq, &k, false, "identity", "Attention_K"},
                  {Ev, &v, false, "identity", "Attention_V"}, "Attention");
    // clang-format on

    Input<> target(A.shape, "target");
    L2Loss<> loss({&A, &target}, "L2Error");

    uint32 bytes = A.numels() * sizeof(FloatT);
    flushing_exec(3, "Attention", [&]() {
        loss.compute();
        loss.backward();
        return bytes;
    });
    return 0;
}

int main(int argc, const char** argv)
{
    std::string name = (argc > 1) ? argv[0] : "main";
    // clang-format off
    std::stringstream usage("\n\nUsage: \n\t");
    usage << name + " time_mult_0             for timing MM with preset sizes\n\t" 
          << name + " time_mult      b h w    for timing A(h,w) * B(w, h) \n\t" 
          << name + " time_mult_2    b h w h2 for timing A(h,w) * B(w, h2)\n\t"
          << name + " time_transpose b h w    for timing transpose A(h,w) \n\t"
          << name + " time_reduce             for timing matrix reduce    \n\t"
          << name + " time_linear_node        for timing linear with bias \n\t"
          << name + " time_attention          for timing attention node \n\t";

    std::map<std::string, int32> commands = {
        {"time_mult_0", 2},
        {"time_mult", 5},
        {"time_mult_2", 6},
        {"time_transpose", 5},
        {"time_reduce", 2},
        {"time_linear_node", 2},
        {"time_attention", 2}
    };

    // clang-format on

    if (argc <= 1 || commands.find(argv[1]) == commands.end() || argc != commands[argv[1]])
    {
        std::stringstream ss;
        for (int32 i = 0; i < argc; i++) ss << argv[i] << " ";
        LOG(RED, "\n\t", usage.str(), RESET, ORANGE, "\nInstead called:\n\t", ss.str().c_str());
        throw_rte_with_backtrace("Invalid usage");
    }

    if (argv[1] == std::string("time_mult") or argv[1] == std::string("time_mult_2"))
    {
        auto A = init_argv(argv);
        uint32 k = (argc > 4) ? strtoul(argv[4], nullptr, 10) : A.width();
        auto B = xavier_uniform_init<FloatT>({A.batch(), A.width(), k});
        LOG(A.shape, " @ ", B.shape, " -> ", A.shape.set(2, B.width()));
        run_mm_timing(A, B);
    }
    else if (argv[1] == std::string("time_mult_0"))
    {
        run_mm_timing();
    }
    else if (argv[1] == std::string("time_attention"))
    {
        time_attention();
    }
    else if (argv[1] == std::string("time_transpose"))
    {
        run_transpose_timing(init_argv(argv));
    }
    else if (argv[1] == std::string("time_reduce"))
    {
        run_reduce_timing();
    }
    else if (argv[1] == std::string("time_linear_node"))
    {
        time_linear_node();
    }

    return 0;
}
