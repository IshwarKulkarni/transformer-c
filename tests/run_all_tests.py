#!/usr/bin/python3

from pathlib import Path
import os
import sys
import time
import subprocess
from datagen import write_sample_mult_data, write_softmax_grad_data

program = ["bin/test"]
passed_tests = []
failed_tests = []

t_colors = {  # terminal colors
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "end": "\033[0m"
}


def run_main(args=[]):
    out = ""
    start = time.time()
    try:
        main_program = [str(x) for x in program + args]
        main_prog_txt = ' '.join(main_program)
        print("Running test: ", main_prog_txt)
        out = subprocess.check_output(main_prog_txt, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(t_colors['red'] + "Test `" + main_prog_txt +
              "`\n Failed with code:" + str(e.returncode) + t_colors['end'])
        print(e.output)
        failed_tests.append(main_prog_txt)
        return False
    passed_tests.append(main_prog_txt)
    print(
        f"{out}Done in {t_colors['blue']} {time.time() - start:2.4f}s {t_colors['end']}\n")


def build(debug=False):
    print("Building " + ("debug" if debug else "release") + " mode")
    build_app = ["make", "-j"] + (["debg=1"] if debug else [])
    try:
        make_result = subprocess.check_output(build_app, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Test failed: ", make_result.returncode)
        print(make_result)
        return False
    return True


def test_mult():
    def test_against_torch(m, n, k=None):
        args = ["test_mult_csv"]
        args += write_sample_mult_data(m, n, k)
        run_main(args)

    sizes = [(5, 12, 12), (8, 24, 24), (2, 1024, 1024), (1, 64, 512, 1),
             (11, 300, 400, 20), (13, 64, 512, 1), (20, 64, 1200, 1)]
    for size in sizes:
        test_against_torch(*(size[1:])) # csv test done with 1 as batch size

    for size in sizes:  # test against C++ implementation
        if len(size) == 4:
            run_main(["test_mult_2"] + list(size))
        else:
            run_main(["test_mult"] + list(size))

    print(t_colors["green"],
          "Matrix multiplication tests passed", t_colors["end"])


def test_softmax_grads():
    sizes = [(1, 24), (5, 17), (9, 512), (4, 65), (20, 30), (30, 40), (32, 300), (300, 15),
             (512, 512), (1024, 1024)]
    for size in sizes:
        csvs = write_softmax_grad_data(*size)
        run_main(["test_softmax_grads"] + csvs)

def test_bin_ops():
    sizes = [(2, 1, 24), (3, 5, 17), (6, 9, 512), (3, 4, 65), (4, 20, 30), 
            (5, 30, 40), (3, 32, 300), (10, 300, 15), (6, 512, 512), (4, 1024, 1024)]
    for size in sizes:
        run_main(["test_bin_ops"] + list(size))


def test_un_ops():
    sizes = [(2, 1, 24), (3, 5, 17), (6, 9, 512), (3, 4, 65), (4, 20, 30), 
            (5, 30, 40), (3, 32, 300), (10, 300, 15), (6, 512, 512), (4, 1024, 1024)]
    for size in sizes:
        run_main(["test_un_ops"] + list(size))

def test_transpose():
    sizes = [(3, 30, 40), (2, 512, 512), (1, 1024, 1024)]
    for size in sizes:
        run_main(["test_transpose"] + list(size))
    print(t_colors["green"],  "Transpose tests passed", t_colors["end"])


def test_reduce():
    sizes = [(4, 1, 24), (6, 5, 17), (5, 1, 512), (6, 4, 65), (5, 20, 30), (7, 30, 40), (8, 32, 300), (16, 300, 15),
             (32, 512, 512), (16, 1024, 1024), (16, 500, 3000)]
    for size in sizes:
        run_main(["test_reduce"] + list(size))
    print(t_colors["green"],  "Reduce tests passed", t_colors["end"])


def time_mult():
    sizes = [(512, 256, 7), (512, 512), (2048, 512), (2048, 2048),
             (2048, 1024, 40), (64, 1200, 1), (4096, 4096)]
    for sizes in sizes:
        selc = ["time_mult"] if len(sizes) == 2 else ["time_mult_2"]
        run_main(selc + list(sizes))


def time_transpose():
    sizes = [(512, 20), (512, 512), (2048, 512)]
    for sizes in sizes:
        run_main(["time_transpose"] + list(sizes))


all_functions = [
    time_mult,
    time_transpose,
    test_mult,
    test_transpose,
    test_reduce,
    test_softmax_grads,
    test_bin_ops, 
    test_un_ops
]

if __name__ == "__main__":
    args = set(sys.argv)
    memcheck = "memcheck" in args
    if memcheck:
        program = ["/usr/local/cuda-12.5/bin/compute-sanitizer", "--leak-check full", "bin/test"]
    build(debug=memcheck)

    test_funcs = [f for f in all_functions if "test" in f.__name__]
    timing_funcs = [f for f in all_functions if "timing" in f.__name__]

    if "all" in args:
        for func in all_functions:
            func()
    elif "tests" in args:
        for func in test_funcs:
            func()
        run_main()
    elif "timing" in args and len(args) == 2:
        for func in timing_funcs:
            func()

    for arg in args:
        for func in all_functions:
            if arg == func.__name__:
                func()

    if len(passed_tests):
        print(t_colors['green'] + "Passed tests: \n",
              '\n'.join(passed_tests), t_colors['end'])
    if len(failed_tests):
        print(t_colors['red'] + "Failed tests: \n", '\n'.join(failed_tests))
        with open("failed_tests.sh", "w", ) as f:
            f.write('\n'.join(failed_tests))

    if len(passed_tests) + len(failed_tests) == 0:
        print("No tests run")
        print("Usage:\n tests/run_all_tests.py <option>\n option is one of\n\t" +
              '\n\t'.join([f.__name__ for f in all_functions] + ["all", "tests", "timing"]))
