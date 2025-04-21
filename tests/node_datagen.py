#! /usr/bin/env python3

# -------------------------------------------------------------------------------
# Author: Ishwar Kulkarni
#
# This file is distributed under the MIT license.
# See: https://mit-license.org
# -------------------------------------------------------------------------------

import torch
import os
import math
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from datagen import save_tensor_to_csv

data_dir = "data/"

def my_assert(a, b, eps=1e-6):
    if (a.shape != b.shape):
        print("shapes mismatch: a.shape", a.shape, 'b.shape: ',  b.shape)
        assert(False)
    if(not torch.allclose(a, b, eps)):
        print("\na:\n", a, "\nb:\n", b, "\na/b:\n", a/b)
        assert(False)

def gen_attention_data():
    """Generate test data for attention mechanism"""
    torch.manual_seed(551)
    torch.set_printoptions(precision=8, linewidth=2000)

    bn = 2
    x0w = 3  # input embedding size
    Eq = 5   # query embedding size
    Ek = Eq  # key embedding size
    Ev = x0w # value, i.e. output embedding size
    S = 7    # seq_len

    Q = torch.nn.Parameter(torch.randn(Eq, x0w) * math.sqrt(2.0/(x0w + Eq)))
    K = torch.nn.Parameter(torch.randn(Ek, x0w) * math.sqrt(2.0/(x0w + Ek)))
    V = torch.nn.Parameter(torch.randn(Ev, x0w) * math.sqrt(2.0/(x0w + Ev)))

    q = torch.rand(bn, S, x0w)
    k = torch.rand(bn, S, x0w)
    v = torch.rand(bn, S, x0w)

    q_ = q @ Q.t()
    k_ = k @ K.t()
    v_ = v @ V.t()

    q_.retain_grad()
    k_.retain_grad()
    v_.retain_grad()

    qkt = q_ @ k_.transpose(1, 2) / (Eq ** .5)
    smax = torch.softmax(qkt, dim=-1)
    output = smax @ v_

    qkt.retain_grad()
    smax.retain_grad()
    output.retain_grad()

    target = torch.randn(output.shape) * 2 + 1
    e = (target - output).pow(2).mean()
    e.backward()

    filename = data_dir + "attention.txt"
    try:
        os.remove(filename)
    except:
        pass

    with open(filename, "w") as f:
        f.write(f"{bn} {x0w} {Eq} {Ek} {Ev} {S} {e.item()}\n")

    l2_grad = 2 * (output - target) / (output.numel())
    qkt_grad = l2_grad @ v_.transpose(1, 2)
    q_grad_in = (qkt.grad @ k_) / (Eq ** .5)
    k_grad_in = (qkt.grad.transpose(1, 2) @ q_) / (Eq ** .5)
    v_grad_in = smax.transpose(1, 2) @ l2_grad

    Q_grad_in = (q_grad_in.transpose(1, 2) @ q)
    K_grad_in = (k_grad_in.transpose(1, 2) @ k)
    V_grad_in = (v_grad_in.transpose(1, 2) @ v)

    my_assert(q_.grad, q_grad_in)
    my_assert(k_.grad, k_grad_in)
    my_assert(v_.grad, v_grad_in)
    my_assert(Q.grad, Q_grad_in.sum(0))
    my_assert(K.grad, K_grad_in.sum(0))
    my_assert(V.grad, V_grad_in.sum(0))

    for target in [Q, K, V, q, k, v, target, qkt, smax, output, q_, k_, v_, Q.grad, K.grad, V.grad]:
        save_tensor_to_csv(target, filename, True)

def gen_linear_data():
    """Generate test data for linear layers"""
    torch.manual_seed(501)
    torch.set_printoptions(precision=8, linewidth=2000, sci_mode=False)
    
    bn = 3
    x0w = 6
    Sl = 4
    I0 = 5
    I2 = 6

    x0 = torch.randn(bn, Sl, x0w)
    W0 = torch.nn.Parameter(torch.randn(I0, x0w))
    W1 = torch.nn.Parameter(torch.randn(I2, I0))
    b1 = torch.nn.Parameter(torch.randn(1, I2))

    z1 = x0 @ W0.t()
    y1 = z1.sigmoid()
    z2 = (y1 @ W1.t() + b1)

    z2.retain_grad()
    y1.retain_grad()
    z1.retain_grad()

    target = torch.randn(z2.shape)
    e = (target - z2).pow(2).mean()
    e.backward()

    filename = data_dir + "linear.txt"
    try:
        os.remove(filename)
    except:
        pass

    with open(filename, "a") as f:
        f.write(f"{bn} {x0w} {Sl} {I0} {I2} {e.item()}\n")

    def sigmoid_backward(x):
        return x * (1 - x)

    z2_grad = 2 * (z2 - target) / z2.numel()
    my_assert(z2.grad, z2_grad)

    y1_grad = z2_grad @ W1
    my_assert(y1.grad, y1_grad)

    w2_grad = z2_grad.transpose(1, 2) @ y1
    my_assert(W1.grad, w2_grad.sum(0))

    b2_grad = z2_grad.sum(dim=1, keepdim=True)
    my_assert(b1.grad, b2_grad.sum(0))

    z1_grad = y1_grad * sigmoid_backward(y1)
    my_assert(z1.grad, z1_grad)

    w1_grad = z1_grad.transpose(1, 2) @ x0
    my_assert(W0.grad, w1_grad.sum(0))

    for tensor in [x0, target, W0, W1, b1, z2, y1, z2.grad, y1.grad, W1.grad, b1.grad, W0.grad]:
        save_tensor_to_csv(tensor, filename, True)

def gen_lsmce_data():
    """Generate test data for log softmax cross entropy loss"""
    torch.manual_seed(10)
    torch.set_printoptions(edgeitems=2000, linewidth=200, sci_mode=False, threshold=2000, precision=10, profile="full")

    x = torch.randn(3, 5, 3)
    W0 = torch.nn.Parameter(torch.randn(5, 3))
    b0 = torch.nn.Parameter(torch.randn(1, 5))

    L = x @ W0.t() + b0
    L.retain_grad()

    target = torch.tensor([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0]
    ]).float()
    target = [target] * x.shape[0]
    target = torch.stack(target)

    o = -target * torch.nn.functional.log_softmax(L, dim=-1)
    e = o.sum(-1).mean()

    e.retain_grad()
    e.backward()

    filename = data_dir + "lsmce.txt"
    try:
        os.remove(filename)
    except:
        pass

    save_tensor_to_csv(x, filename, True)
    save_tensor_to_csv(W0, filename, True)
    save_tensor_to_csv(b0, filename, True)
    save_tensor_to_csv(target, filename, True)

    with open(filename, "a") as f:
        f.write(f"{e.item()}\n")

    save_tensor_to_csv(L, filename, True)
    save_tensor_to_csv(L.grad, filename, True)
    save_tensor_to_csv(W0.grad, filename, True)
    save_tensor_to_csv(b0.grad, filename, True)

def gen_softmax_dim_data():
    """Generate test data for softmax along different dimensions"""
    torch.set_printoptions(precision=8, linewidth=2000, sci_mode=False)
    torch.manual_seed(1331)

    bn = 2
    Ei = 68
    Sl = 33
    Eq = 35

    def write_softmax(N=1):
        x0 = torch.randn(bn, Sl, Ei)
        W0 = torch.nn.Parameter(torch.randn(Eq, Ei))
        b0 = torch.nn.Parameter(torch.randn(1, Eq))
        z1 = (x0 @ W0.t() + b0).tanh()
        s = torch.softmax(z1, dim=(-1-N))    
        t = torch.randn(s.shape)
        e = (-t * s.log()).mean()
        
        z1.retain_grad()
        s.retain_grad()
        e.retain_grad()
        e.backward()
        
        filename = data_dir + f"sm_dim{N}.txt"
        try:
            os.remove(filename)
        except:
            pass

        with open(filename, "a") as f:
            f.write(f"{bn} {Ei} {Sl} {Eq} {e.item()}\n")

        for t in [x0, W0, b0, t, z1, s, z1.grad, W0.grad, b0.grad]:
            save_tensor_to_csv(t, filename, True)
        
        #print(f"z1.grad.abs().sum(): {z1.grad.abs().sum()}")
        #print(f"W0.grad.abs().sum(): {W0.grad.abs().sum()}")
        #print(f"b0.grad.abs().sum(): {b0.grad.abs().sum()}\n")

    write_softmax(0)
    write_softmax(1)

def gen_average_data():
    """Generate test data for average operation"""
    torch.manual_seed(999)
    torch.set_printoptions(precision=8, linewidth=2000, sci_mode=False)

    bn = 3
    Ei = 7
    Sl = 4
    Eq = 5

    x0 = torch.randn(bn, Sl, Ei)
    W0 = torch.nn.Parameter(torch.randn(Eq, Ei))
    b0 = torch.nn.Parameter(torch.randn(1, Eq))
    z1 = (x0 @ W0.t() + b0).tanh()

    y1 = z1.mean(dim=1, keepdim=True)
    y1.retain_grad()
    target = torch.randn(y1.shape)
    e = (target - y1).pow(2).mean()
    e.backward()

    filename = data_dir + "average.txt"
    try:
        os.remove(filename)
    except:
        pass

    with open(filename, "w") as f:
        f.write(f"{bn} {Ei} {Sl} {Eq} {e.item()}\n")

    for t in [x0, W0, b0, target, z1, y1, W0.grad, b0.grad]:
        save_tensor_to_csv(t, filename, True)

def gen_layer_norm_data():
    """Generate test data for layer normalization"""
    torch.manual_seed(99)
    torch.set_printoptions(precision=8, linewidth=2000, sci_mode=False)

    bn = 2
    Ei = 5
    Sl = 3
    Eq = 6

    x0 = torch.randn(bn, Sl, Ei) + 0.5
    W0 = torch.nn.Parameter(torch.randn(Eq, Ei))
    b0 = torch.nn.Parameter(torch.randn(1, Eq))
    y = (x0 @ W0.t() + b0).sigmoid()

    norm = torch.nn.LayerNorm(y.shape[-1])
    z = norm(y)
    z.retain_grad()
    y.retain_grad()

    y_mean = y.mean(dim=-1, keepdim=True)
    y_sq = y ** 2
    y_sq_mean = y_sq.mean(dim=-1, keepdim=True)
    y_var = y_sq_mean - y_mean ** 2
    y_num = y - y_mean
    y_std = y_var ** 0.5
    y_norm = y_num/y_std

    target = torch.randn(z.shape)
    e = (target - y_norm).pow(2).mean()
    e.backward()

    my_assert(y_norm, z, 1e-3)

    filename = data_dir + "layer_norm.txt"
    try:
        os.remove(filename)
    except:
        pass

    with open(filename, "w") as f:
        f.write(f"{bn} {Ei} {Sl} {Eq} {e.item()}\n")

    for t in [x0, W0, b0, target, y, z, W0.grad, b0.grad]:
        save_tensor_to_csv(t, filename, True)

def gen_product_data():
    """Generate test data for product operation"""
    torch.manual_seed(509)
    torch.set_printoptions(precision=8, linewidth=2000, sci_mode=False)

    bn = 5
    Ei = 17
    Sl = 14
    Eq = 15
    I2 = 16
    I3 = 17

    x0 = torch.randn(bn, Sl, Ei)
    W0 = torch.nn.Parameter(torch.randn(Eq, Ei))
    b0 = torch.nn.Parameter(torch.randn(1, Eq))

    x1 = torch.randn(bn, I3, I2)
    W1 = torch.nn.Parameter(torch.randn(Eq, I2))

    z0 = x0 @ W0.t() + b0
    y0 = z0.sigmoid()

    y1 = x1 @ W1.t()
    A = y0 @ y1.transpose(1, 2) / 2.222

    t = torch.randn(A.shape)
    e = (t - A).pow(2).mean()

    A.retain_grad()
    y1.retain_grad()
    y0.retain_grad()
    z0.retain_grad()

    e.backward()

    filename = data_dir + "productT.txt"
    try:
        os.remove(filename)
    except:
        pass

    with open(filename, "w") as f:
        f.write(f"{bn} {Ei} {Sl} {Eq} {I2} {I3} {e.item()}\n")

    for tensor in [x0, W0, b0, x1, W1, t, A, y1, y0, A.grad, W0.grad, b0.grad, W1.grad]:
        save_tensor_to_csv(tensor, filename, True)

def gen_division_data():
    """Generate test data for division operation"""
    torch.manual_seed(509)

    bn = 5
    Ei = 17
    Sl = 14
    Eq = 15

    x0 = torch.randn(bn, Sl, Ei)
    W0 = torch.nn.Parameter(torch.randn(Eq, Ei))

    x1 = torch.randn(bn, Sl, Ei)
    W1 = torch.nn.Parameter(torch.randn(Eq, Ei))

    z0 = x0 @ W0.t()
    y0 = z0.sigmoid()

    y1 = x1 @ W1.t()
    y1 = y1.sigmoid()
    att_out = y0 / y1
    t = torch.randn(att_out.shape)
    e = (t - att_out).pow(2).mean()

    att_out.retain_grad()
    y1.retain_grad()
    y0.retain_grad()
    z0.retain_grad()

    e.backward()

    diff_grad = 2 * (att_out - t) / att_out.numel()
    my_assert(att_out.grad, diff_grad)
    my_assert(y0.grad, diff_grad / y1)
    my_assert(y1.grad, -diff_grad * y0 / (y1 ** 2))

def gen_self_attention_data():
    """Generate test data for self attention"""
    torch.manual_seed(510)
    torch.set_printoptions(precision=8, linewidth=2000)

    bn = 4
    x0w = 3  # input embedding size
    Eq = 6   # query embedding size
    S = 8    # seq_len

    I = torch.nn.Parameter(torch.randn(x0w, x0w))
    Q = torch.nn.Parameter(torch.randn(Eq, x0w) * math.sqrt(2.0/(x0w + Eq)))
    K = torch.nn.Parameter(torch.randn(Eq, x0w) * math.sqrt(2.0/(x0w + Eq)))
    V = torch.nn.Parameter(torch.randn(Eq, x0w) * math.sqrt(2.0/(x0w + Eq)))

    x = torch.rand(bn, S, x0w)
    x_ = (x @ I.t())

    q_ = x_ @ Q.t()
    k_ = x_ @ K.t()
    v_ = x_ @ V.t()

    q_.retain_grad()
    k_.retain_grad()
    v_.retain_grad()

    qkt = q_ @ k_.transpose(1, 2) / (Eq ** .5)
    smax = torch.softmax(qkt, dim=-1)
    output = smax @ v_

    qkt.retain_grad()
    smax.retain_grad()
    output.retain_grad()

    target = torch.randn(output.shape) * 2 + 1
    e = (target - output).pow(2).mean()
    e.backward()

    filename = data_dir + "self_attention.txt"
    try:
        os.remove(filename)
    except:
        pass

    with open(filename, "w") as f:
        f.write(f"{bn} {x0w} {Eq} {S} {e.item()}\n")

    for tensor in [x, I, Q, K, V, target, x_, q_, k_, v_, qkt, smax, output, Q.grad, K.grad, V.grad, I.grad]:
        save_tensor_to_csv(tensor, filename, True)

def gen_cross_attention_data():
    """Generate test data for cross attention"""
    torch.manual_seed(51)
    torch.set_printoptions(precision=8, linewidth=2000)

    bn = 4
    x0w = 3  # input embedding size
    Eq = 6   # query embedding size
    S = 8    # seq_len

    Iq = torch.nn.Parameter(torch.randn(x0w, x0w))
    Ikv = torch.nn.Parameter(torch.randn(x0w, x0w))

    Q = torch.nn.Parameter(torch.randn(Eq, x0w) * math.sqrt(2.0/(x0w + Eq)))
    K = torch.nn.Parameter(torch.randn(Eq, x0w) * math.sqrt(2.0/(x0w + Eq)))
    V = torch.nn.Parameter(torch.randn(Eq, x0w) * math.sqrt(2.0/(x0w + Eq)))

    x = torch.rand(bn, S, x0w)
    x_q = (x @ Iq.t()).tanh()
    x_kv = (x @ Ikv.t()).tanh()

    q_ = x_q @ Q.t()
    k_ = x_kv @ K.t()
    v_ = x_kv @ V.t()

    q_.retain_grad()
    k_.retain_grad()
    v_.retain_grad()

    qkt = q_ @ k_.transpose(1, 2) / (Eq ** .5)
    smax = torch.softmax(qkt, dim=-1)
    output = smax @ v_

    qkt.retain_grad()
    smax.retain_grad()
    output.retain_grad()

    target = torch.randn(output.shape) * 2 + 1
    e = (target - output).pow(2).mean()
    e.backward()

    filename = data_dir + "cross_attention.txt"
    try:
        os.remove(filename)
    except:
        pass

    with open(filename, "w") as f:
        f.write(f"{bn} {x0w} {Eq} {S} {e.item()}\n")

    for tensor in [x, Iq, Ikv, Q, K, V, target, x_q, x_kv, q_, k_, v_, qkt, smax, output, Q.grad, K.grad, V.grad, Iq.grad, Ikv.grad]:
        save_tensor_to_csv(tensor, filename, True)

def gen_adam_data():
    """Generate test data for adam optimizer"""
    torch.manual_seed(511)
    torch.set_printoptions(precision=8, linewidth=2000)

    bn = 4
    def gaussian_2d(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta=0):
        """
        2D Gaussian function.
        """
        a = np.cos(theta)**2 / (2 * x_stddev**2) + np.sin(theta)**2 / (2 * y_stddev**2)
        b = -np.sin(2 * theta) / (4 * x_stddev**2) + np.sin(2 * theta) / (4 * y_stddev**2)
        c = np.sin(theta)**2 / (2 * x_stddev**2) + np.cos(theta)**2 / (2 * y_stddev**2)
        exponent = a * (x - x_mean)**2 + 2 * b * (x - x_mean) * (y - y_mean) + c * (y - y_mean)**2
        return amplitude * np.exp(-exponent)

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

    g0, g1 = np.gradient(Z)
    g = (g1**2 + g0**2) ** 0.5


    if False:
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(Z1, cmap='viridis')
        ax[1].imshow(Z2, cmap='viridis')
        ax[2].imshow(Z3, cmap='viridis')
        ax[3].imshow(Z4, cmap='viridis')

        plt.show()
        plt.imshow(Z, cmap='terrain')
        plt.colorbar()
        plt.show()

        plt.imshow(g, cmap='viridis')
        plt.colorbar()

    save_tensor_to_csv(torch.Tensor(Z), "data/adam_v.csv")
    save_tensor_to_csv(torch.Tensor(g0), "data/adam_g0.csv")
    save_tensor_to_csv(torch.Tensor(g1), "data/adam_g1.csv")
    

if __name__ == "__main__":
    functions = {name: obj for name, obj in globals().items() 
                if callable(obj) and obj.__module__ == __name__ and name.startswith("gen_")}
    if len(sys.argv) == 1 or  (len(sys.argv) > 1 and sys.argv[1] == "all"):
        for name in functions.keys():
            print(f"Generating {name}")
            functions[name]()
    else:
        num_generated = 0
        for name in sys.argv[1:]:
            if name in functions.keys():
                functions[name]()
                num_generated += 1
        if num_generated == 0:
            all_names = "\n\t".join(functions.keys())
            print("No functions generated, pass no arg or one of:\n", all_names)
