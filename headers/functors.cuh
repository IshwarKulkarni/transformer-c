#ifndef FUNCTOR_CUH
#define FUNCTOR_CUH

#include "types"
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

//////////////////////////////////////////////////////////////////////////////////////
// Unary functors
//////////////////////////////////////////////////////////////////////////////////////
template <typename T> struct Identity
{
    inline __host__ __device__ T operator()(T a) const { return a; }
};

template <typename T> struct Neg
{
    inline __host__ __device__ T operator()(const T x) const { return -x; }
};

template <typename Ta> struct Square
{
    __host__ __device__ inline Ta operator()(Ta a) const { return a * a; }
};

template <typename Ta> struct Exp
{
    __host__ __device__ inline Ta operator()(Ta a) const { return exp(a); }
};

template <typename Ta> struct Loge
{
    __host__ __device__ inline Ta operator()(Ta a) const { return log(a); }
};

template <typename Ta> struct Abs
{
    __host__ __device__ inline Ta operator()(Ta a) const { return abs(a); }
};

template <typename Ta> struct Sign
{
    __host__ __device__ inline Ta operator()(Ta a) const { return (a > 0) ? 1 : -1; }
};

template <typename Ta> struct Sqrt
{
    __host__ __device__ inline Ta operator()(Ta a) const { return sqrt(a); }
};

template <typename T> struct DividebBy
{
    T divisor;
    DividebBy(T divisor) : divisor(divisor) {}
    __host__ __device__ inline T operator()(T a) const { return a / divisor; }
};

//////////////////////////////////////////////////////////////////////////////////////
// Binary functors
//////////////////////////////////////////////////////////////////////////////////////

template <typename Ta, typename Tb = Ta> struct Sub
{
    static constexpr Ta Identity = 0;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a - b; }
};

template <typename Ta, typename Tb = Ta> struct Mul
{
    static constexpr Ta Identity = 1;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a * b; }
};

template <typename Ta, typename Tb = Ta> struct Plus
{
    static constexpr Ta Identity = 0;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a + b; }
};

template <typename T> struct Max
{
    static constexpr T Identity = std::numeric_limits<T>::lowest();
    __host__ __device__ inline T operator()(T a, T b) const { return (a > b ? a : b); }
};

template <typename T> struct Min
{
    static constexpr T Identity = std::numeric_limits<T>::max();
    __host__ __device__ inline T operator()(T a, T b) const { return (a <= b ? a : b); }
};

//////////////////////////////////////////////////////////////////////////////////////
// Activations (have a forward and backward Unary functors)
//////////////////////////////////////////////////////////////////////////////////////

template <typename Ta> struct TanhF
{
    __host__ __device__ inline Ta operator()(Ta a) const { return tanh(a); }
};

template <typename Ta> struct TanhB
{
    __host__ __device__ inline Ta operator()(Ta a) const { return 1 - a * a; }
};
template <typename Ta> struct ReluF
{
    __host__ __device__ inline Ta operator()(Ta a) const { return a > 0 ? a : 0; }
};

template <typename Ta> struct ReluB
{
    __host__ __device__ inline Ta operator()(Ta a) const { return a > 0 ? 1 : 0; }
};

/////// Activation structs

template <typename T> struct Sigmoid
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

template <typename T> struct Tanh
{
    TanhF<T> forward;
    TanhB<T> backward;
};

template <typename T> struct Relu
{
    ReluF<T> forward;
    ReluB<T> backward;
};

template <typename T> struct IdentityActivation
{
    Identity<T> forward;
    Identity<T> backward;
};
//////////////////////////////////////////////////////////////////////////////////////
// composition of unary operators
//////////////////////////////////////////////////////////////////////////////////////

// mostly for the fancyness, it's shorter/easier to write new functors
template <typename T, typename F, typename... Rest>
struct Composition // chaining left to right, F(rest(...))
{
    F f;
    Composition<T, Rest...> rest;
    __host__ __device__ inline T operator()(T a) const { return rest(f(a)); }

    __host__ __device__ inline T operator()(T a, T b) const { return rest(f(a, b)); }
};

template <typename T, typename F> struct Composition<T, F> // last functor
{
    F f;
    __host__ __device__ inline T operator()(T a) const { return f(a); }

    __host__ __device__ inline T operator()(T a, T b) const { return f(a, b); }
};

template <typename T> // (a - b)^2
using DiffSq = Composition<T, Sub<T>, Square<T>>;

template <typename T> // sign(a - b)
using DiffSign = Composition<T, Sub<T>, Sign<T>>;

template <typename T, int32 Multiplier> struct IntegerMultiplier
{
    __host__ __device__ inline T operator()(T a) const { return a * Multiplier; }
};

#endif
