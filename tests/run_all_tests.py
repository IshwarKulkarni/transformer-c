#!/usr/bin/python3.8

from pathlib import Path
import sys
import subprocess

program = ["bin/main"]

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
        print("Running test: \n" + term_colors['yellow']  + main_prog_txt + term_colors['end'])
        out = subprocess.check_output(main_prog_txt, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(term_colors['red'] + "Test `" + main_prog_txt + "`\n Failed with code:" + str(e.returncode) + term_colors['end'])
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

def write_sample_mult_data(height, width, height2=None, data_path=Path('./temp')):
    import torch
    height2 = height if height2 is None else height2
    print(f"Writing sample data for matrix multiplication: {height}x{width} @ {width}x{height2}")
    a = torch.rand(height, width)
    b = torch.rand(width, height2)

    def save_tensor_to_csv(tensor, filename):
        with open(filename, 'w') as f:
            f.write(f"{tensor.shape[0]} {tensor.shape[1]}\n")
            for i in range(tensor.shape[0]):
                f.write(' '.join([str(x) for x in tensor[i].tolist()]) + '\n')

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

    sizes = [(12, 12), (24, 24), (1024, 1024), (300, 400, 20), (64, 512, 1), (64, 1200, 1)]
    for size in sizes:
        test_csv_size(*size)

    print( term_colors["green"] ,  "Matrix multiplication tests passed", term_colors["end"])

def transpose_tests():
    print("Running transpose tests")
    sizes = [(30, 40), (512, 512), (1024, 1024)]
    for size in sizes:
        run_main(["test_transpose"] + list(size))
    print( term_colors["green"] ,  "Transpose tests passed", term_colors["end"])

def mult_timing():
    print("Running matrix multiplication timing")
    sizes = [(512, 256, 7), (512, 512), (2048, 512), (2048, 2048), (2048, 1024, 40), (64, 1200, 1) , (4096, 4096)]
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
    "test_transpose": transpose_tests
}

def main():
    args = set(sys.argv)
    global program
    memcheck = "memcheck" in args
    if memcheck:
        program = ["cuda-memcheck", "--leak-check full", "bin/main"]
    build(debug=memcheck)
   
    if "all" in args:
        for func in test_time_functions.values():
            func()
        return

    allowed_args = ["tests", "time", "transpose", "mult"]
    args = [x for x in args if x in allowed_args]
    for arg in args:
        for name, func in test_time_functions.items():
            if arg in name:
                func()
        

if __name__ == "__main__":
   main()