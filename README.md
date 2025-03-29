# Transformer-C

A C++ implementation of the Transformer architecture with CUDA support for efficient matrix operations.

## Project Structure

```
.
├── bin/           # Compiled executables
├── data/          # Test data and Python comparison files
├── headers/       # Header files
├── notebooks/     # Jupyter notebooks for comparison and training
├── src/           # Source files
├── static_data/   # Static data files
└── tests/         # Test files
```

## Building the Project

The project uses a Makefile for building. The following commands are available:

```bash
# Build all executables
make build

# Build with debug symbols
make build dbg=1

# Clean build files
make clean

# Run specific executable with variables
make run_<executable_name> var="<arguments>"

# Run with Valgrind for memory checking
make valgrind
```

## Main Components

### Matrix Operations

The project implements efficient matrix operations with CUDA support:

- Matrix multiplication
- Matrix addition/subtraction
- Matrix transpose
- Element-wise operations
- Various matrix transformations

### Node Class Hierarchy

The project implements a node-based computational graph with the following operations:

- Linear/Dense layers
- Attention Layer
- Layer normalization
    - Batch, layer and sequence normalization
- Dropout
- Compound nodes like:
    - Feed-forward networks
    - Multi-head attention
- Softmax activation
- Matrix multiplication
- Element-wise addition/subtraction
- Tensor reshaping and transposition

### Network Builder and Serialization

The project provides a network description language and serialization capabilities through the NetworkBuilder class:

#### Network Description Language

Networks can be defined using a simple text-based format that describes the architecture in a hierarchical manner. The format supports:

```text
#Linear: linear1
    out_dim: 160
    prev: input1
    use_bias: true 
    activation: relu

$rate: 0.2

Dropout: dropout1
    prev: linear1
    rate: $rate

Input: target
    batch: $batch_size
    height: input1->height
    width: sa1->out_dim

L2Loss: l2loss
    target: target
    predictions: dropout2

```

The language supports:
- Layer definitions with parameters
- Nested layer structures
- Network-wide configurations
- Comments and documentation
- Variable substitution
- Layer reuse through references

#### Weight Serialization

The project provides comprehensive weight serialization capabilities:

```cpp
// Saving weights
NetworkBuilder builder;
builder.load_network("network_config.txt");
builder.save_weights("model_weights.bin");

// Loading weights
NetworkBuilder builder;
builder.load_network("network_config.txt");
builder.load_weights("model_weights.bin");
```

Weight serialization features:
- Binary format for efficient storage
- Support for partial weight updates
- Version control for backward compatibility

The weight file format includes:
- Header with version and metadata
- Layer-wise weight organization
- Training statistics and metrics

## Testing and Validation

The project includes a comprehensive test suite to validate the implementation of various operations and optimizations. The test infrastructure consists of two main components:

### Test Data Generation (`tests/datagen.py`)

This script generates test data for various operations:

```bash
python tests/datagen.py <gen_command> <args>
```

Available commands:
- `gen_data_mult`: Generate data for matrix multiplication
- `gen_data_transpose`: Generate data for transpose operations
- `gen_data_reduce_sum/min/max`: Generate data for reduce operations
- `gen_data_softmax_grads`: Generate data for softmax gradient testing
- `gen_data_adam`: Generate data for Adam optimizer testing

Example usage:
```bash
python tests/datagen.py gen_data_mult 5 12 12  # Generate data for 5x12 @ 12x12 matrix multiplication
python tests/datagen.py gen_data_softmax_grads 1 24  # Generate data for 1x24 softmax gradients
```

### Test Runner (`tests/run_all_tests.py`)

The test runner provides a flexible way to execute different categories of tests:

```bash
python tests/run_all_tests.py <option>
```

Available options:
- `all`: Run all tests including timing tests
- `tests`: Run only validation tests (no timing)
- `timing`: Run only timing tests
- Individual test functions:
  - `test_mult`: Matrix multiplication tests
  - `test_transpose`: Transpose operation tests
  - `test_reduce`: Reduce operation tests
  - `test_softmax_grads`: Softmax gradient tests
  - `test_bin_ops`: Binary operation tests
  - `test_un_ops`: Unary operation tests
  - `time_mult`: Matrix multiplication timing tests
  - `time_transpose`: Transpose operation timing tests

Additional options:
- `memcheck`: Run tests with CUDA compute sanitizer for memory leak detection
- `debug`: Build in debug mode (automatically enabled with memcheck)

Example usage:
```bash
# Run all tests
python tests/run_all_tests.py all

# Run only validation tests
python tests/run_all_tests.py tests

# Run specific test with memory checking
python tests/run_all_tests.py test_mult memcheck
```

The test suite validates:
- Matrix operations (multiplication, transpose)
- Reduction operations (sum, min, max)
- Neural network operations (softmax, gradients)
- Binary and unary operations
- Optimizer implementations (Adam)
- Memory management and CUDA operations

Failed tests are logged to `failed_tests.sh` for easy rerunning of failed tests.

## Performance Optimization

- CUDA acceleration for matrix operations
- Memory optimization for large matrices
- Efficient tensor operations
- Support for batch processing

## Dependencies

- C++17 compatible compiler
- CUDA toolkit (tested with CUDA 12.5)
- Python 3.x (for comparison notebooks)
- PyTorch (for comparison)

## Usage Examples

```cpp
// Matrix multiplication example
Matrix<float> A(64, 512);
Matrix<float> B(512, 1);
Matrix<float> C = A * B;

// Node-based computation example
Node* input = new InputNode(data);
Node* output = new LinearNode(input, weights);
output->forward();
```

## Development

- Code formatting is handled by `.clang-format`
- Compilation database is available in `compile_commands.json`
- Debug builds can be enabled with `dbg=1` make flag

## TODO:

- Visualizing weights and Softmax output
- Move training to network builder and add to serialization.
- Add ways to set optiomizer over all params.


