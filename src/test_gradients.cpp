#include "../headers/matrix_ops.hpp"

static constexpr float64 pi = 3.14159265358979323846;

inline int test_sampling()
{
    Matrixf values(200, 200, "values");

    auto sinc = [](float64 y, float64 x) {
        if (x == 0 and y == 0) return 1.0;
        auto v = std::sqrt(x * x + y * y);
        return std::sin(v) / v;
    };

    for (uint32 y = 0; y < values.height; y++)
    {
        float64 y_ = y / (float64)values.height;
        y_ *= pi;
        y_ -= pi / 2;
        for (uint32 x = 0; x < values.width; x++)
        {
            float64 x_ = x / (float64)values.width;
            x_ *= pi;
            x_ -= pi / 2;
            auto v = sinc(y_, x_);
            values(y, x) = v;
        }
    }

    Matrixf grads(values.shape(), "grads");
    fillCPU(grads, -1.);
    for (uint32 y = 0; y < values.height; y++)
    {
        float64 y_ = y / (float64)values.height;
        for (uint32 x = 0; x < values.width; x++)
        {
            float64 x_ = x / (float64)values.width;
            auto [gy, gx] = gradient_xy(values, y_, x_);
            grads(y, x) = std::sqrt(gx * gx + gy * gy);
        }
    }

    write_csv(values, "values.csv");
    write_csv(grads, "grads.csv");
    return 0;
}

void test_sin_cos()
{
    auto width = 314;
    Matrixf vals(1, width, "x");
    Matrixf sine(1, width, "sin");
    Matrixf cos(1, width, "cos");
    Matrixf grads(1, width, "grads");
    for (uint32 x = 0; x < width; x++)
    {
        auto x_ = x / (float64)width;
        vals(0, x) = x_;
        sine(0, x) = std::sin(x_);
        cos(0, x) = std::cos(x_);
    }

    for (uint32 x = 0; x < vals.width; x++)
    {
        auto x_ = x / (float64)width;
        auto [gy, gx] = gradient_xy(sine, 0, x_, 1e-3);
        LOG("x: ", x, " x_: ", x_, " val: ", vals(0, x), " sine: ", sine(0, x), " cos: ", cos(0, x),
            " gx: ", gx);
        grads(0, x) = gx;
    }

    write_csv(sine, "sine.csv");
    write_csv(cos, "cos.csv");
    write_csv(grads, "grads.csv");
}

void test_sq_cube()
{
    auto width = 314;
    Matrixf vals(1, width, "x");
    Matrixf cube(1, width, "cube");
    Matrixf square(1, width, "square");
    Matrixf grads(1, width, "grads");
    for (uint32 x = 0; x < width; x++)
    {
        auto x_ = x / (float64)width;
        x_ = (x_ - 0.5) * 3;
        vals(0, x) = x_;
        cube(0, x) = x_ * x_ * x_;
        square(0, x) = x_ * x_;
    }

    for (uint32 x = 0; x < vals.width; x++)
    {
        auto x_ = x / (float64)width;
        auto [gy, gx] = gradient_xy(cube, 0, x_, 1e-1);
        // gx /= 9;
        LOG("x: ", x, " x_: ", x_, " val: ", vals(0, x), " cube: ", cube(0, x),
            " square: ", square(0, x), " gx: ", gx, " gx/sqaure: ", gx / square(0, x));
        grads(0, x) = gx;  /// square(0, x);
    }

    write_csv(cube, "cube.csv");
    write_csv(square, "square.csv");
    write_csv(grads, "grads.csv");
}

void test_poly()
{
    auto width = 314;
    Matrixf vals(1, width, "x");
    Matrixf polynomial(1, width, "poly");         // 3x^2 + 4x + sin(x)
    Matrixf polynomial_diff(1, width, "square");  // 6x + 4 + cos(x)
    Matrixf grads(1, width, "grads");
    for (uint32 x = 0; x < width; x++)
    {
        auto x_ = x / (float64)width;
        x_ = (x_ - 0.5) * 2;
        vals(0, x) = x_;
        auto sin = std::sin(x_);
        polynomial(0, x) = 3 * x_ * x_ + 4 * x_ + sin;
        polynomial_diff(0, x) = 6 * x_ + 4 + std::cos(x);
    }

    for (uint32 x = 0; x < vals.width; x++)
    {
        auto x_ = x / (float64)width;
        auto [gy, gx] = gradient_xy(polynomial, 0, x_, 1e-1);
        LOG("x: ", x, " x_: ", x_, " val: ", vals(0, x), " poly: ", polynomial(0, x),
            " poly_diff: ", polynomial_diff(0, x), " gx: ", gx,
            " gx/sqaure: ", gx / polynomial_diff(0, x));
        grads(0, x) = gx;  /// square(0, x);
    }

    write_csv(polynomial, "poly.csv");
    write_csv(polynomial_diff, "poly_diff.csv");
    write_csv(grads, "grads.csv");
}

void test_tan_sec2()
{
    auto width = 314;
    Matrixf vals(1, width, "x");
    Matrixf tan(1, width, "tan");
    Matrixf sec2(1, width, "sec2");
    Matrixf grads(1, width, "grads");
    for (uint32 x = 0; x < width; x++)
    {
        auto x_ = x / (float64)width;
        x_ = (x_ - 0.5) * (pi - 0.1);
        vals(0, x) = x_;
        tan(0, x) = std::tan(x_);
        sec2(0, x) = 1 / (std::cos(x_) * std::cos(x_));
    }

    for (uint32 x = 0; x < vals.width; x++)
    {
        auto x_ = x / (float64)width;
        auto [gy, gx] = gradient_xy(tan, 0, x_, 1e-3);
        gx /= 3.044;
        LOG("x: ", x, " x_: ", x_, " val: ", vals(0, x), " tan: ", tan(0, x), " sec2: ", sec2(0, x),
            " gx: ", gx, " gx/sec2: ", gx / sec2(0, x));
        grads(0, x) = gx;
    }

    write_csv(tan, "tan.csv");
    write_csv(sec2, "sec2.csv");
    write_csv(grads, "grads.csv");
}
