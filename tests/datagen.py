#!/usr/bin/python3
import torch
import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import math
from pathlib import Path

data_path = Path('./data')
os.makedirs(data_path, exist_ok=True)


def save_tensor_to_csv(tensor, filename, append=False):
    with open(filename, 'a' if append else 'w') as f:
        f.write(f"{tensor.shape[0]} {tensor.shape[1]}\n")
        for i in range(tensor.shape[0]):
            f.write(' '.join([str(x) for x in tensor[i].tolist()]) + '\n')
    return filename


def write_sample_reduce_data(height, width, op):

    print(f"Writing sample data for reduce operation: {height}x{width} @ {op}")
    a = torch.rand(height, width)
    if op == "sum":
        result = torch.sum(a, dim=1, keepdim=True)
    elif op == "min":
        result = torch.min(a, dim=1, keepdim=True)[0]
    elif op == "max":
        result = torch.max(a, dim=1, keepdim=True)[0]
    else:
        raise ValueError("Invalid operation")
    save_tensor_to_csv(a, data_path/'a.csv')
    save_tensor_to_csv(result, data_path/'result.csv')


def write_softmax_grad_data(height, width):
    a = torch.nn.Parameter(torch.randn(height, width))
    t = torch.nn.Parameter(torch.randn(height, width))

    torch.set_printoptions(precision=10)
    s = torch.softmax(a, 0)
    s.retain_grad()
    mse = torch.nn.MSELoss()(s, t)
    mse.retain_grad()
    mse.backward()

    save_tensor_to_csv(s,      'data/s_out.csv')
    save_tensor_to_csv(s.grad, 'data/s_grad_in.csv')
    save_tensor_to_csv(a.grad, 'data/s_grad_out.csv')
    return ['data/s_out.csv', 'data/s_grad_in.csv', 'data/s_grad_out.csv']


def write_sample_mult_data(height, width, height2=None):
    import torch
    height2 = height if height2 is None else height2
    print(
        f"Writing sample data for matrix multiplication: {height}x{width} @ {width}x{height2}")
    a = torch.rand(height, width)
    b = torch.rand(width, height2)
    c = torch.mm(a, b)

    filenames = [
        save_tensor_to_csv(a, data_path/f'a_{a.shape[0]}x{a.shape[1]}.csv'),
        save_tensor_to_csv(b, data_path/f'b_{a.shape[0]}x{a.shape[1]}.csv'),
        save_tensor_to_csv(c, data_path/f'c{c.shape[0]}x{c.shape[1]}.csv')]

    return [str(p) for p in filenames]

def gen_data_adam_heightmap(size):
    
    def gaussian_2d(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta=0):
        a = np.cos(theta)**2 / (2 * x_stddev**2) + np.sin(theta)**2 / (2 * y_stddev**2)
        b = -np.sin(2 * theta) / (4 * x_stddev**2) + np.sin(2 * theta) / (4 * y_stddev**2)
        c = np.sin(theta)**2 / (2 * x_stddev**2) + np.cos(theta)**2 / (2 * y_stddev**2)
        exponent = a * (x - x_mean)**2 + 2 * b * (x - x_mean) * (y - y_mean) + c * (y - y_mean)**2
        return amplitude * np.exp(-exponent)

    np.random.seed(0)
    x_lin = np.linspace(-1, 1, 400)
    y_lin = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x_lin, y_lin)

    Z1 = gaussian_2d(X, Y, 1.0, -.75, .5, .50, 2, math.pi/4)
    Z2 = gaussian_2d(X, Y, 0.5, -.2, .5, .4, 2, -math.pi/4) * 2
    Z3 = gaussian_2d(X, Y, 1.0, .75, 0.15, .25, 3, math.pi/3) * -.5
    Z4 = gaussian_2d(X, Y, 1.0, 1.05, 0.2, .25, 3, math.pi/3) * -.5

    r = np.random.randn(*Z1.shape) * 2
    r = gaussian_filter(r, 12)
    Z = Z1 + Z2 + Z3 + Z4 + r

    #g0, g1 = np.gradient(Z)
    #g = (g1**2 + g0**2) ** 0.5
    save_tensor_to_csv(torch.Tensor(Z), "static_data/adam_v.csv")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python datagen.py <gen_command> <args>")

    height = int(sys.argv[2])
    width = int(sys.argv[3])

    if sys.argv[1] == "gen_data_mult":
        write_sample_mult_data(height, width, int(sys.argv[4]))
    elif sys.argv[1] == "gen_data_transpose":
        write_sample_mult_data(height, width)
    elif sys.argv[1] == "gen_data_reduce_sum":
        write_sample_reduce_data(height, width, "sum")
    elif sys.argv[1] == "gen_data_reduce_min":
        write_sample_reduce_data(height, width, "min")
    elif sys.argv[1] == "gen_data_reduce_max":
        write_sample_reduce_data(height, width, "max")
    elif sys.argv[1] == "gen_data_softmax_grad":
        write_softmax_grad_data(height, width)
    elif sys.argv[1] == "gen_data_adam":
        if(height != width):
            print("Height and width must be equal for adam heightmap")
        gen_data_adam_heightmap(height, height)
    else:
        raise ValueError("Invalid command")
