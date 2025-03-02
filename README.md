# SimpleLogisticRegression

## Overview

SimpleLogisticRegression is a C++ implementation of logistic regression with selectable loss functions (L1/L2) and activation functions (Sigmoid/ReLU). It uses Eigen for matrix operations and supports training on a dataset from a CSV file, followed by prediction on a validation dataset.

## Features

- Supports **L1 (Lasso) and L2 (Ridge) regularization**
- Allows selection of **Sigmoid or ReLU activation functions**
- Uses **gradient descent** for model optimization
- Reads training and validation datasets from CSV files
- Outputs predictions to a CSV file

## Requirements

- C++14 or later
- Eigen3 library

### Installing Eigen3 (Linux)

```sh
sudo apt-get install libeigen3-dev
```

### Installing Eigen3 (Mac)

```sh
brew install eigen
```

For Windows, download Eigen from [Eigen Official Site](https://eigen.tuxfamily.org/) and extract it into your project include directory.

## Directory Structure

```
.
├── train.csv
├── validate.csv
├── logistic_regression.cpp
├── predict_output.csv (Generated after running the program)
└── CMakeLists.txt (Optional for CMake build)
```

## Example CSV Files

### `train.csv`

```csv
feature1,feature2,truth
1.0,2.0,1
2.0,1.0,0
3.0,4.0,1
4.0,3.0,0
5.0,6.0,1
```

### `validate.csv`

```csv
feature1,feature2
2.0,3.0
5.0,1.0
4.0,4.0
```

## Build and Run

### Using CMake

```sh
mkdir build
cd build
cmake ..
make
./logistic_regression
```

### Using g++ (Without CMake)

```sh
g++ -std=c++14 logistic_regression.cpp -o logistic_regression -I /path/to/eigen3
./logistic_regression
```

## Usage

Modify the following parameters in `logistic_regression.cpp` before running:

```cpp
string lossFunction = "L2";      // Choose "L1" or "L2"
string activationFunction = "sigmoid";  // Choose "sigmoid" or "relu"
double alpha = 0.1;             // Learning rate
int iterations = 1000;          // Number of iterations
double lambda = 0.1;            // Regularization strength
```

## Output

The program generates `predict_output.csv` containing predictions:

```csv
feature1,feature2,prediction
2.0,3.0,1
5.0,1.0,0
4.0,4.0,1
```

## License

This project is open-source and available under the MIT License.
