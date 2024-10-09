#!/usr/bin/env python
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

def write_out_data(m, n, write_mul_result=False):
    a = torch.rand(m, n)
    b = torch.rand(n, m)

    def save_tensor_to_csv(tensor, filename):
        with open(filename, 'w') as f:
            f.write(f"{tensor.shape[0]} {tensor.shape[1]}\n")
            for i in range(tensor.shape[0]):
                f.write(' '.join([str(x) for x in tensor[i].tolist()]) + '\n')

    save_tensor_to_csv(a, f'data/a.csv')
    save_tensor_to_csv(b, f'data/b.csv')
    if write_mul_result:
        c = a @ b
        save_tensor_to_csv(c, f'data/c.csv')

def run_test():
    
    sizes = [ [1024, 1024], [2000, 256], [1024, 512], [512, 2000], [2048, 64], [64, 2000], [600, 1200]]
    sizes = sorted(sizes, key=lambda x: x[0] * x[1])
    print("Testing for sizes", sizes)
    def get_time(m, n):
        os.system(f"./bin/main {m} {n} > output.txt")

        #read output.txt and collate it
        with open("output.txt", "r") as f:
            line = [l for l in f.readlines() if l.startswith("--Timing ")][0]
            return float(line.split("|")[1])

    with open("timing_results.txt", "w") as resutlt_file:
        for m, n in sizes:
            time = get_time(m, n)
            print(f"{8} {m}x{n}\t\t {m * n} \t {time} ", file=resutlt_file)
            if m != n:
                time = get_time(n, m)
                print(f"{8} {n}x{m}\t\t {m * n } \t {time} ", file=resutlt_file)
            resutlt_file.flush()

def plot():

    # Step 1: Read the data
    data = pd.read_csv('timing_results.txt', delim_whitespace=True, header=None, names=['block_size', 'matrix_size', 'num_elements', 'time_taken'])

    # Step 2: Group the data by block size
    grouped = data.groupby('block_size')

    # Step 3: Plot the data
    plt.figure()
    for name, group in grouped:
        plt.plot(group['num_elements'], group['time_taken'], label=f'Block Size {name}')
        
    #Step3.1: Add dot for the data points and label them with the matrix size
    for i, row in data.iterrows():
        plt.scatter(row['num_elements'], row['time_taken'])
        label = row['matrix_size']
        x, y = label.split('x')
        plt.text(row['num_elements'], row['time_taken'], f"{x}x{y}")

    # Step 4: Set log-log scale
    plt.xscale('log')
    plt.yscale('log')

    # Step 5: Add labels and legend
    plt.xlabel('Number of Elements')
    plt.ylabel('Time Taken')
    plt.legend()

    # Step 6: Save the plot as a PNG file
    plt.savefig('plot.png', dpi=1200)
    plt.show()

if __name__ == "__main__":
    run_test()
    plot()