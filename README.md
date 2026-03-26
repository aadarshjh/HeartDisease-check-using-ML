# Heart Disease Prediction using Machine Learning

A machine learning project that predicts the presence of heart disease in a patient using Logistic Regression.

## Overview

This project uses a Logistic Regression classifier trained on clinical patient data to predict whether a person has heart disease. The model is built with scikit-learn and evaluated on a standard heart disease dataset.

## Dataset

The dataset (`heart.csv`) contains **1025 patient records** with **13 clinical features** and a binary target variable.

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiographic results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia type (0–3) |
| `target` | **Target** — 1 = Heart Disease, 0 = No Heart Disease |

## Project Structure

```
HeartDisease-check-using-ML/
├── MachineLearning.py   # Main script for training and prediction
├── heart.csv            # Heart disease dataset
└── README.md            # Project documentation
```

## Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install numpy pandas scikit-learn
```

## How to Run

```bash
python MachineLearning.py
```

## How It Works

1. **Load data** — Reads `heart.csv` into a pandas DataFrame.
2. **Split features & target** — Separates the 13 input features from the `target` column.
3. **Train/test split** — 80% training, 20% testing (stratified, `random_state=2`).
4. **Feature scaling** — Applies `StandardScaler` to normalise features.
5. **Train model** — Fits a `LogisticRegression` classifier (`max_iter=1000`).
6. **Evaluate** — Prints training and test accuracy scores.
7. **Predict** — Runs inference on a sample patient input and prints the result.

## Example Output

```
Accuracy on Training data: 0.85
Accuracy on Test data: 0.82
The Person has Heart Disease
```

## Results

The Logistic Regression model achieves ~82–85% accuracy on this dataset, demonstrating that basic clinical measurements are effective predictors of heart disease risk.

## License

This project is open-source and available for educational use.
