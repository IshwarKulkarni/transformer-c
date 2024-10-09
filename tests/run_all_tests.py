#!/usr/bin/python3.8

from pathlib import Path
import os
import sys
import time
import subprocess
from datagen import write_sample_mult_data

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
    print("Building project: " + ("debug" if debug else "release"))
    subprocess.check_output(["make", "clean"], shell=True, text=True)
    build_app = ["make"] + (["debg=1"] if debug else [])
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

    sizes = [(12, 12), (24, 24), (1024, 1024), (1, 5, 1),
             (300, 400, 20), (64, 512, 1), (64, 1200, 1)]
    for size in sizes:
        test_against_torch(*size)

    for size in sizes:  # test against C++ implementation
        if len(size) == 3:
            run_main(["test_mult_2"] + list(size))
        else:
            run_main(["test_mult"] + list(size))

    print(t_colors["green"],
          "Matrix multiplication tests passed", t_colors["end"])


def test_transpose():
    sizes = [(30, 40), (512, 512), (1024, 1024)]
    for size in sizes:
        run_main(["test_transpose"] + list(size))
    print(t_colors["green"],  "Transpose tests passed", t_colors["end"])


def test_reduce():
    sizes = [(30, 40), (32, 300), (300, 15),
             (512, 512), (1024, 1024), (500, 3000)]
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
    test_reduce
]

if __name__ == "__main__":
    args = set(sys.argv)
    memcheck = "memcheck" in args
    if memcheck:
        program = ["cuda-memcheck", "--leak-check full", "bin/test"]
    build(debug=memcheck)

    test_funcs = [f for f in all_functions if "test" in f.__name__]
    timing_funcs = [f for f in all_functions if "timing" in f.__name__]

    if "all" in args:
        for func in all_functions:
            func()
    elif "tests" in args:
        for func in test_funcs:
            func()
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
