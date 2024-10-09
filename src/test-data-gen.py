#%%
import torch
import sys

m = int(sys.argv[1])
n = int(sys.argv[2])

a = torch.rand(m, n)
b = torch.rand(n, m)

def save_tensor_to_csv(tensor, filename):
    with open(filename, 'w') as f:
        f.write(f"{tensor.shape[0]} {tensor.shape[1]}\n")
        for i in range(tensor.shape[0]):
            f.write(' '.join([str(x) for x in tensor[i].tolist()]) + '\n')

save_tensor_to_csv(a, f'data/a.csv')
save_tensor_to_csv(b, f'data/b.csv')
c = a @ b
save_tensor_to_csv(c, f'data/c.csv')