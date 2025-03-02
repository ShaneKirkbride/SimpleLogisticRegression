#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
using namespace Eigen;

// Function to read CSV file into Eigen matrix
MatrixXd readCSV(const string& filename, bool hasHeader = false) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    if (hasHeader && file.good()) {
        getline(file, line); // Skip header
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }

        data.push_back(row);
    }

    MatrixXd mat(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        mat.row(i) = VectorXd::Map(&data[i][0], data[i].size());
    }

    return mat;
}

// Function to save predictions to a CSV file
void saveCSV(const string& filename, const MatrixXd& features,
             const VectorXd& predictions) {
    ofstream file(filename);
    file << "feature1,feature2,prediction\n"; // Adding header
    for (int i = 0; i < features.rows(); ++i) {
        file << features(i, 0) << "," << features(i, 1) << "," << predictions(i)
             << "\n";
    }
    file.close();
}

// Sigmoid activation function
VectorXd sigmoid(const VectorXd& z) {
    return 1.0 / (1.0 + (-z.array()).exp());
}

// ReLU activation function
VectorXd relu(const VectorXd& z) {
    return z.array().max(0.0);
}

// Function to select activation function dynamically
VectorXd activate(const VectorXd& z, const string& activationFunction) {
    if (activationFunction == "sigmoid") {
        return sigmoid(z);
    } else if (activationFunction == "relu") {
        return relu(z);
    } else {
        cerr << "Unknown activation function: " << activationFunction
             << ". Use 'sigmoid' or 'relu'." << endl;
        exit(1);
    }
}

// Logistic regression using gradient descent with L1 or L2 loss and selectable
// activation function
VectorXd logisticRegression(const MatrixXd& X, const VectorXd& y, double alpha,
                            int iterations, const string& lossFunction = "L2",
                            const string& activationFunction = "sigmoid",
                            double lambda = 0.1) {
    int m = X.rows();
    int n = X.cols();
    VectorXd theta = VectorXd::Zero(n);

    for (int i = 0; i < iterations; ++i) {
        VectorXd predictions = activate(X * theta, activationFunction);
        VectorXd errors = predictions - y;

        if (lossFunction == "L2") {
            // L2 (Ridge) Regularization Gradient Descent
            VectorXd gradient = (X.transpose() * errors) / m + lambda * theta;
            gradient(0) -=
                lambda * theta(0); // Exclude bias term from regularization
            theta -= alpha * gradient;
        } else if (lossFunction == "L1") {
            // L1 (Lasso) Regularization Gradient Descent
            VectorXd signTheta = theta.array().sign(); // Sign function for L1
            VectorXd gradient =
                (X.transpose() * errors) / m + lambda * signTheta;
            gradient(0) -=
                lambda * signTheta(0); // Exclude bias term from regularization
            theta -= alpha * gradient;
        } else {
            cerr << "Unknown loss function: " << lossFunction
                 << ". Use 'L1' or 'L2'." << endl;
            exit(1);
        }
    }

    return theta;
}

// Main function
int main() {
    // Load training data
    MatrixXd trainData = readCSV("train.csv", true);
    MatrixXd X_train = trainData.leftCols(trainData.cols() - 1);
    VectorXd y_train = trainData.rightCols(1);

    // Add a bias term (column of ones) to the feature matrix
    MatrixXd X_train_bias(X_train.rows(), X_train.cols() + 1);
    X_train_bias << MatrixXd::Ones(X_train.rows(), 1), X_train;

    // Choose hyperparameters
    string lossFunction = "L2"; // "L1" or "L2" loss function
    string activationFunction =
        "sigmoid";         // "sigmoid" or "relu" activation function
    double alpha = 0.1;    // Learning rate
    int iterations = 1000; // Number of iterations
    double lambda = 0.1;   // Regularization strength

    // Train the logistic regression model
    VectorXd theta =
        logisticRegression(X_train_bias, y_train, alpha, iterations,
                           lossFunction, activationFunction, lambda);

    // Load validation data
    MatrixXd X_validate = readCSV("validate.csv", true);

    // Add bias term to validation data
    MatrixXd X_validate_bias(X_validate.rows(), X_validate.cols() + 1);
    X_validate_bias << MatrixXd::Ones(X_validate.rows(), 1), X_validate;

    // Predict using the logistic regression model
    VectorXd predictions =
        activate(X_validate_bias * theta, activationFunction);
    predictions =
        (predictions.array() >= 0.5).cast<double>(); // Convert to binary output

    // Save predictions to a CSV file
    saveCSV("predict_output.csv", X_validate, predictions);

    cout << "Predictions saved to predict_output.csv" << endl;

    return 0;
}
