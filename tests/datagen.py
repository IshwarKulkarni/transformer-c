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

    if sys.argv[1] == "gen_data_mult":
        write_sample_mult_data(int(sys.argv[2]), int(
            sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "gen_data_transpose":
        write_sample_mult_data(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "gen_data_reduce_sum":
        write_sample_reduce_data(int(sys.argv[2]), int(sys.argv[3]), "sum")
    elif sys.argv[1] == "gen_data_reduce_min":
        write_sample_reduce_data(int(sys.argv[2]), int(sys.argv[3]), "min")
    elif sys.argv[1] == "gen_data_reduce_max":
        write_sample_reduce_data(int(sys.argv[2]), int(sys.argv[3]), "max")
