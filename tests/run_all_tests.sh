#!/bin/bash 

set -euxo pipefail
set -x

make clean;
make > /dev/null

bin/main test_transpose 512 300

python tests/run_datagen.py 512 512
bin/main test_mult_csv temp/*.csv

python tests/run_datagen.py 300 400
bin/main test_mult_csv temp/*.csv

bin/main test_transpose 512 512 
bin/main test_transpose 300 300 


bin/main time_mult 512 512
bin/main time_mult 2048 512

bin/main time_transpose 512 512
bin/main time_transpose 2048 512
