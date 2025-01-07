#ifndef FUNCTOR_CUH
#define FUNCTOR_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include "types"

//////////////////////////////////////////////////////////////////////////////////////
// Unary functors
//////////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct Identity
{
    inline __host__ __device__ T operator()(T a) const { return a; }
    static constexpr char const* name = "Identity";
};

template <typename T>
struct Neg
{
    inline __host__ __device__ T operator()(const T a) const { return -a; }
    static constexpr char const* name = "Unary Negation";
};

template <typename Ta>
struct Square
{
    __host__ __device__ inline Ta operator()(Ta a) const { return a * a; }
};

template <typename Ta>
struct Exp  // can apply shfted value
{
    Ta shift;
    Exp(Ta shift = 0) : shift(shift) {}
    __host__ __device__ inline Ta operator()(Ta a) const { return exp(a - shift); }
    static constexpr char const* name = "Exp";
};

template <typename Ta>
struct Loge
{
    __host__ __device__ inline Ta operator()(Ta a) const { return log(a); }
};

template <typename Ta>
struct Abs
{
    __host__ __device__ inline Ta operator()(Ta a) const { return abs(a); }
};

template <typename Ta>
struct Sign
{
    Ta multiplier = 1;
    Sign(Ta multiplier = 1) : multiplier(multiplier) {}
    __host__ __device__ inline Ta operator()(Ta a) const
    {
        return (a > 0) ? multiplier : -multiplier;
    }
};

template <typename Ta>
struct Sqrt
{
    __host__ __device__ inline Ta operator()(Ta a) const { return sqrt(a); }
};

template <typename T>
struct DividedBy
{
    T divisor;
    DividedBy(T divisor) : divisor(divisor) {}
    __host__ __device__ inline T operator()(T a) const { return a / divisor; }
    static constexpr char const* name = "DividedBy";
};

template <typename T>
struct MultiplyBy
{
    T factor;
    MultiplyBy(T factor) : factor(factor) {}
    __host__ __device__ inline T operator()(T a) const { return a * factor; }
};

template <typename T, int32 Multiplier>
struct IntegerMultiplier
{
    __host__ __device__ inline T operator()(const T a) const { return a * Multiplier; }
};

template <typename T, int32 Multiplier>
struct IntegerDivider
{
    __host__ __device__ inline T operator()(T a) const { return a / Multiplier; }
};

//////////////////////////////////////////////////////////////////////////////////////
// Binary functors
//////////////////////////////////////////////////////////////////////////////////////

template <typename Ta, typename Tb = Ta>
struct Plus
{
    static constexpr Ta Identity = 0;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a + b; }
    static constexpr char const* name = "Plus";
};

template <typename Ta, typename Tb = Ta>
struct Sub
{
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a - b; }
    static constexpr char const* name = "Sub";
};

template <typename Ta, typename Tb = Ta>
struct Mul
{
    static constexpr Ta Identity = 1;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a * b; }
    static constexpr char const* name = "Mul";
};

template <typename Ta, typename Tb = Ta>
struct Div
{
    const Tb epsilon = Tb(1e-6);
    static constexpr Ta Identity = 1;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a / (b + epsilon); }
};

template <typename Ta, typename Tb = Ta>
struct NegLogLossBckwd
{
    Ta normalizing_factor = 1;
    const Tb epsilon = Tb(1e-6);
    static constexpr Ta Identity = Ta(0);
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const
    {
        return (-a / (b + epsilon)) / normalizing_factor;
    }
};

template <typename T, typename Act>
struct ActBackwardMul
{
    using actBackT = typename Act::backward;
    const typename Act::backward actBack = {};
    __host__ __device__ inline T operator()(T output, T inGrad) const
    {
        return actBack(output) * inGrad;
    }
};

template <typename T>
struct LSMCEBkwd
{
    // dL/dxi = exp(-nls) - t :  where, t is target and nls = negLogSoftmax
    // and  -nls = [xi - log(Sum(e^xj))],
    // exp(-nls) =  e^xi/(Sum(e^xj))
    T factor = 1;
    __host__ __device__ inline T operator()(T t, T nls) const { return (exp(-nls) - t) / factor; }
};

template <typename T>
struct NLSToSoftmax
{
    // dL/dxi = exp(-nls) - t :  where, t is targeta and nls = negLogSoftmax
    // and  -nls = [xi - log(Sum(e^xj))],
    // exp(-nls) =  e^xi/(Sum(e^xj))
    __host__ __device__ inline T operator()(T nls) const { return exp(-nls); }
};

template <typename Ta, typename Tb = Ta>
struct NegLogLossFwd
{
    Ta normalizing_factor = 1;
    __host__ __device__ inline Ta operator()(Ta t, Tb o) const
    {
        return -t * log(o) / normalizing_factor;
    }
};

template <typename WT, typename WuT = WT>
struct WeightUpdate
{
    static constexpr WT Identity = 0;
    WuT clip = 1;
    float64 learning_rate;
    WeightUpdate(float64 learning_rate = 1e-3) : learning_rate(learning_rate) {}
    __host__ __device__ inline WT operator()(WT a, WuT b) const { return a - learning_rate * b; }
};

template <typename WT, typename WuT = WT>
struct ClippedWeightUpdate
{
    WuT clip = 0.25;
    WT learning_rate;
    ClippedWeightUpdate(WT learning_rate) : learning_rate(learning_rate) {}
    __host__ __device__ inline WT operator()(WT a, WuT b) const
    {
        auto clipped_b = b > clip ? clip : b < -clip ? -clip : b;
        return a - learning_rate * clipped_b;
    }
};

template <typename WT>
struct MomentUpdate
{
    const float32 beta1;
    MomentUpdate(float64 beta1) : beta1(beta1) {}
    __host__ __device__ inline WT operator()(WT m, WT grad) const
    {
        return beta1 * m + (1.f - beta1) * grad;
    }
};

template <typename WT>
struct SecondMomentUpdate
{
    const float32 beta2;
    SecondMomentUpdate(float64 beta2) : beta2(beta2) {}
    __host__ __device__ inline WT operator()(WT v, WT grad) const
    {
        return beta2 * v + (1.f - beta2) * grad * grad;
    }
};

template <typename WT>
struct AdamWeightUpdate
{
    const float32 beta1d;  // beta1 decayed
    const float32 beta2d;  // beta2 decayed
    const float32 epsilon = 1e-8;

    AdamWeightUpdate(float32 b1d, float32 b2d, float32 eps = 1e-8)
        : beta1d(b1d), beta2d(b2d), epsilon(eps)
    {
    }

    __host__ __device__ inline WT operator()(WT m, WT v) const
    {
        if (abs(1 - beta1d) < epsilon || abs(1 - beta2d) < epsilon)
        {
            return m / (sqrt(v) + epsilon);
        }
        float32 m_hat = m / (1.0 - beta1d);
        float32 v_hat = v / (1.0 - beta2d);
        float32 out = m_hat / (sqrt(v_hat) + epsilon);
        return WT(out);
    }
};

template <typename T>
struct Max
{
    static constexpr T Identity = std::numeric_limits<T>::lowest();
    __host__ __device__ inline T operator()(T a, T b) const { return (a > b ? a : b); }
    static constexpr char const* name = "Max";
};

template <typename T>
struct Min
{
    static constexpr T Identity = 1e9;
    __host__ __device__ inline T operator()(T a, T b) const { return (a <= b ? a : b); }
    static constexpr char const* name = "Min";
};

//////////////////////////////////////////////////////////////////////////////////////
// Activations (have a forward and backward Unary functors)
//////////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct Sigmoid
{
    typedef struct SigmoidF
    {
        __host__ __device__ inline T operator()(T a) const { return 1 / (1 + T(exp(-a))); }
        static constexpr char const* name = "SigmoidForward";
    } forward;

    typedef struct SigmoidB
    {
        __host__ __device__ inline T operator()(T a) const { return a * (1 - a); }
        static constexpr char const* name = "SigmoidBackward";
    } backward;
    static constexpr char const* name = "Sigmoid";
};

template <typename T>
struct Relu
{
    typedef struct ReluF
    {
        __host__ __device__ inline T operator()(T a) const { return a > 0 ? a : 0; }
        static constexpr char const* name = "ReluForward";
    } forward;

    typedef struct ReluB
    {
        __host__ __device__ inline T operator()(T a) const { return a > 0 ? 1 : 0; }
        static constexpr char const* name = "ReluBackward";
    } backward;
};

template <typename T>
struct LeakyRelu
{
    float32 slope;
    typedef struct LeakyReluF
    {
        float32 slope;
        LeakyReluF(float32 negative_slope = 3e-3) : slope(negative_slope) {}
        __host__ __device__ inline T operator()(T a) const { return a > 0 ? a : a * slope; }
        static constexpr char const* name = "LeakyReluForward";
    } forward;

    typedef struct LeakyReluB
    {
        float32 slope;
        LeakyReluB(float32 negative_slope = 3e-3) : slope(negative_slope) {}
        __host__ __device__ inline T operator()(T a) const { return a > 0 ? 1 : slope; }
        static constexpr char const* name = "LeakyReluBackward";
    } backward;

    LeakyRelu(float32 neg_slope = 3e-3) : forward(neg_slope), backward(slope) {}
};

template <typename T>
struct TanH
{
    typedef struct TanhF
    {
        __host__ __device__ inline T operator()(T a) const { return tanh(a); }
        static constexpr char const* name = "TanHForward";
    } forward;

    typedef struct TanhB
    {
        __host__ __device__ inline T operator()(T a) const { return 1 - a * a; }
        static constexpr char const* name = "TanHBackward";
    } backward;
    static constexpr char const* name = "TanH";
};

template <typename T>
struct IActivation
{
    typedef Identity<T> forward;
    typedef Identity<T> backward;
    static constexpr char const* name = "IActivation";
};
//////////////////////////////////////////////////////////////////////////////////////
// composition of unary operators
//////////////////////////////////////////////////////////////////////////////////////

// mostly for the fancyness, it's shorter/easier to write new functors
template <typename T, typename F, typename... Rest>
struct Composition  // chaining left to right, F(rest(...))
{
    Composition() = default;
    F f;
    Composition<T, Rest...> rest;
    __host__ __device__ inline T operator()(T a) const { return rest(f(a)); }

    __host__ __device__ inline T operator()(T a, T b) const
    {
        return rest(f(a, b));
    }  // unary( binary(a,b) )
};

template <typename T, typename F>
struct Composition<T, F>  // last functor
{
    F f;
    __host__ __device__ inline T operator()(T a) const { return f(a); }

    __host__ __device__ inline T operator()(const T a, const T b) const { return f(a, b); }
};

template <typename T>  // (a - b)^2
using DiffSq = Composition<T, Sub<T>, Square<T>>;

template <typename T>  // (a - b)
struct MultNeg2
{
    __host__ __device__ inline T operator()(T a, T b) const { return 2 * (a - b); }
};

#endif
