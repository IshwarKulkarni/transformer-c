cuda-memcheck --leak-check full bin/test test_reduce 32 300
cuda-memcheck --leak-check full bin/test test_reduce 512 512
cuda-memcheck --leak-check full bin/test test_reduce 1024 1024
cuda-memcheck --leak-check full bin/test test_reduce 500 3000