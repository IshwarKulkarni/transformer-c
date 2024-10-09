#!/usr/bin/python3.8

from pathlib import Path
import sys
import subprocess

program = ["bin/test"]

term_colors = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "end": "\033[0m"
}


def run_main(args=[]):
    out = ""
    try:
        main_program = [str(x) for x in program + args]
        main_prog_txt = ' '.join(main_program)
        print("Running test: \n" +
              term_colors['yellow'] + main_prog_txt + term_colors['end'])
        out = subprocess.check_output(main_prog_txt, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(term_colors['red'] + "Test `" + main_prog_txt +
              "`\n Failed with code:" + str(e.returncode) + term_colors['end'])
        print(e.output)
        return False
    print(out)


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


def save_tensor_to_csv(tensor, filename):
    with open(filename, 'w') as f:
        f.write(f"{tensor.shape[0]} {tensor.shape[1]}\n")
        for i in range(tensor.shape[0]):
            f.write(' '.join([str(x) for x in tensor[i].tolist()]) + '\n')


def write_sample_reduce_data(height, width, op, data_path=Path('./temp')):
    import torch
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


def write_sample_mult_data(height, width, height2=None, data_path=Path('./temp')):
    import torch
    height2 = height if height2 is None else height2
    print(
        f"Writing sample data for matrix multiplication: {height}x{width} @ {width}x{height2}")
    a = torch.rand(height, width)
    b = torch.rand(width, height2)

    save_tensor_to_csv(a, data_path/'a.csv')
    save_tensor_to_csv(b, data_path/'b.csv')
    save_tensor_to_csv(a @ b, data_path/'c.csv')

    return [str(p) for p in [data_path/'a.csv', data_path/'b.csv', data_path/'c.csv']]


def mult_tests():
    print("Running matrix multiplication tests")

    def test_csv_size(m, n, k=None):
        args = ["test_mult_csv"]
        args += write_sample_mult_data(m, n, k)
        run_main(args)

    sizes = [(12, 12), (24, 24), (1024, 1024),
             (300, 400, 20), (64, 512, 1), (64, 1200, 1)]
    for size in sizes:
        test_csv_size(*size)

    for size in sizes:
        if len(size) == 3:
            run_main(["test_mult_2"] + list(size))
        else:
            run_main(["test_mult"] + list(size))

    print(term_colors["green"],
          "Matrix multiplication tests passed", term_colors["end"])


def transpose_tests():
    print("Running transpose tests")
    sizes = [(30, 40), (512, 512), (1024, 1024)]
    for size in sizes:
        run_main(["test_transpose"] + list(size))
    print(term_colors["green"],  "Transpose tests passed", term_colors["end"])


def reduce_tests():
    print("Running reduce tests")
    sizes = [(30, 40), (32, 300), (300, 15),
             (512, 512), (1024, 1024), (500, 3000)]
    for size in sizes:
        run_main(["test_reduce"] + list(size))
    print(term_colors["green"],  "Reduce tests passed", term_colors["end"])


def mult_timing():
    print("Running matrix multiplication timing")
    sizes = [(512, 256, 7), (512, 512), (2048, 512), (2048, 2048),
             (2048, 1024, 40), (64, 1200, 1), (4096, 4096)]
    for sizes in sizes:
        selc = ["time_mult"] if len(sizes) == 2 else ["time_mult_2"]
        run_main(selc + list(sizes))


def transpose_timing():
    print("Running transpose timing")
    sizes = [(512, 20), (512, 512), (2048, 512)]
    for sizes in sizes:
        run_main(["time_transpose"] + list(sizes))


test_time_functions = {
    "time_mult": mult_timing,
    "time_transpose": transpose_timing,
    "test_mult_csv": mult_tests,
    "test_mult": mult_tests,
    "test_transpose": transpose_tests,
    "test_reduce": reduce_tests
}


def tests_timing():
    args = set(sys.argv)
    global program
    memcheck = "memcheck" in args
    if memcheck:
        program = ["cuda-memcheck", "--leak-check full", "bin/test"]
    build(debug=memcheck)

    if "all" in args or len(args) == 1:
        for func in test_time_functions.values():
            func()
        return

    test_time_functions.get(sys.argv[1], lambda: print("Invalid test"))()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        tests_timing()
    elif sys.argv[1].startswith("gen_data"):
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
    else:
        tests_timing()
