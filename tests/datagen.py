#!/usr/bin/python3
import torch
import sys
import os
from pathlib import Path

data_path = Path('./data')
os.makedirs(data_path, exist_ok=True)


def save_tensor_to_csv(tensor, filename):
    with open(filename, 'w') as f:
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
    else:
        raise ValueError("Invalid command")
