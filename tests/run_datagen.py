#!/usr/bin/env python
import torch
import os
import pandas as pd
from pathlib import Path
import sys

folder = Path("./temp")

os.makedirs(folder, exist_ok=True)

def write_sample_mult_data(m, n, k =None, write_mul_result=False):
    k = m if k is None else k
    a = torch.rand(m, n)
    b = torch.rand(n, k)

    def save_tensor_to_csv(tensor, filename):
        with open(folder/filename, 'w') as f:
            f.write(f"{tensor.shape[0]} {tensor.shape[1]}\n")
            for i in range(tensor.shape[0]):
                f.write(' '.join([str(x) for x in tensor[i].tolist()]) + '\n')

    save_tensor_to_csv(a, f'a.csv')
    save_tensor_to_csv(b, f'b.csv')
    if write_mul_result:
        c = a @ b
        save_tensor_to_csv(c, f'c.csv')

def run_timing():
    sizes = [ [1024, 1024], [2000, 256], [1024, 512], [512, 2000], [2048, 64], [64, 2000], [600, 1200]]
    sizes = sorted(sizes, key=lambda x: x[0] * x[1])
    print("Testing for sizes", sizes)
    def get_time(m, n):
        os.system(f"./bin/main {m} {n} > output.txt")

        #read output.txt and collate it
        with open(folder/"output.txt", "r") as f:
            line = [l for l in f.readlines() if l.startswith("--Timing ")][0]
            return float(line.split("|")[1])

    with open(folder/"timing_results.txt", "w") as resutlt_file:
        for m, n in sizes:
            time = get_time(m, n)
            print(f"{8} {m}x{n}\t\t {m * n} \t {time} ", file=resutlt_file)
            if m != n:
                time = get_time(n, m)
                print(f"{8} {n}x{m}\t\t {m * n } \t {time} ", file=resutlt_file)
            resutlt_file.flush()

def plot():
    import matplotlib.pyplot as plt
    data = pd.read_csv(folder/'timing_results.txt', delim_whitespace=True, header=None, 
                       names=['block_size', 'matrix_size', 'num_elements', 'time_taken'])

    grouped = data.groupby('block_size')

    plt.figure()
    for name, group in grouped:
        plt.plot(group['num_elements'], group['time_taken'], label=f'Block Size {name}')
        
    for i, row in data.iterrows():
        plt.scatter(row['num_elements'], row['time_taken'])
        label = row['matrix_size']
        x, y = label.split('x')
        plt.text(row['num_elements'], row['time_taken'], f"{x}x{y}")

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Number of Elements')
    plt.ylabel('Time Taken')
    plt.legend()

    plt.savefig('plot.png', dpi=1200)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        run_timing()
        plot()
    else:
        k = int(sys.argv[3]) if len(sys.argv) > 3 else None
        write_sample_mult_data(int(sys.argv[1]), int(sys.argv[2]), k, True)
