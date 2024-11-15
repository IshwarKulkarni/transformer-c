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
};

template <typename T>
struct Neg
{
    inline __host__ __device__ T operator()(const T a) const { return -a; }
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
struct DividebBy
{
    T divisor;
    DividebBy(T divisor) : divisor(divisor) {}
    __host__ __device__ inline T operator()(T a) const { return a / divisor; }
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
};

template <typename Ta, typename Tb = Ta>
struct Sub
{
    static constexpr Ta Identity = 0;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a - b; }
};

template <typename Ta, typename Tb = Ta>
struct Mul
{
    static constexpr Ta Identity = 1;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a * b; }
};

template <typename Ta, typename Tb = Ta>
struct Div
{
    static constexpr Ta Identity = 1;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a / b; }
};

template <typename Ta, typename Tb = Ta>
struct NegDiv
{
    static constexpr Ta Identity = 1;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return -a / b; }
};

template <typename Ta, typename Tb = Ta>
struct CrossEntropy
{
    __host__ __device__ inline Ta operator()(Ta t, Tb o) const { return -t * log(o); }
};

template <typename WT, typename WuT = WT>
struct WeightUpdate
{
    static constexpr WT Identity = 0;
    WT learning_rate;
    WeightUpdate(WT learning_rate) : learning_rate(learning_rate) {}
    __host__ __device__ inline WT operator()(WT a, WuT b) const { return a - learning_rate * b; }
};

template <typename T>
struct Max
{
    static constexpr T Identity =
        std::is_floating_point<T>::value ? -1e6 : std::numeric_limits<T>::lowest();
    __host__ __device__ inline T operator()(T a, T b) const { return (a > b ? a : b); }
};

template <typename T>
struct Min
{
    static constexpr T Identity =
        std::is_floating_point<T>::value ? 1e6 : std::numeric_limits<T>::max();
    __host__ __device__ inline T operator()(T a, T b) const { return (a <= b ? a : b); }
};

//////////////////////////////////////////////////////////////////////////////////////
// Activations (have a forward and backward Unary functors)
//////////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct Sigmoid
{
    typedef struct SigmoidF
    {
        __host__ __device__ inline T operator()(T a) const { return 1 / (1 + exp(-a)); }
    } forward;

    typedef struct SigmoidB
    {
        __host__ __device__ inline T operator()(T a) const { return a * (1 - a); }
    } backward;
};

template <typename T>
struct Relu
{
    typedef struct ReluF
    {
        __host__ __device__ inline T operator()(T a) const { return a > 0 ? a : 0; }
    } forward;

    typedef struct ReluB
    {
        __host__ __device__ inline T operator()(T a) const { return a > 0 ? 1 : 0; }
    } backward;
};

template <typename T>
struct TanH
{
    typedef struct TanhF
    {
        __host__ __device__ inline T operator()(T a) const { return tanh(a); }
    } forward;

    typedef struct TanhB
    {
        __host__ __device__ inline T operator()(T a) const { return 1 - a * a; }
    } backward;
};

template <typename T>
struct GELU
{
    typedef struct GELUF
    {
        __host__ __device__ inline T operator()(T a) const
        {
            return 0.5 * a * (1 + tanh(sqrt(2 / M_PI) * (a + 0.044715 * a * a * a)));
        }
    } forward;

    typedef struct GELUB
    {
        __host__ __device__ inline T operator()(T a) const
        {
            T t = tanh(0.0356774 * a * a * a);
            return 0.5 * (1 + t + a * (1 - t * t));
        }
    } backward;
};

template <typename T>
struct IdentityActivation
{
    typedef Identity<T> forward;
    typedef Identity<T> backward;
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

template <typename T>
struct FunctorName
{
    static const char *name() { return "Unknown"; }
};

template <typename T>
struct FunctorName<Mul<T>>
{
    static const char *name() { return "Mul"; }
};
template <typename T>
struct FunctorName<Plus<T>>
{
    static const char *name() { return "Plus"; }
};
template <typename T>
struct FunctorName<DividebBy<T>>
{
    static const char *name() { return "DivBy"; }
};
template <typename T>
struct FunctorName<WeightUpdate<T>>
{
    static const char *name() { return "WeightUpdate"; }
};

#endif
