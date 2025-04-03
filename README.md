# HOIP Prediction Tool

This is a deep learning model with executable python scripts which uses the model to predict the spin splitting.

## Installation

First, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the prediction using the following command:

```bash
python predict.py --model MODEL_PATH --input INPUT_PATH --output OUTPUT_PATH
```

Example:

```bash
python predict.py --model ./best_model.pth --input ./X.csv --output predictions.csv
```

Here, X.csv is our example input file indicating the format we use.

### Parameters

- `--model`: PyTorch model file path (.pth format)
- `--input`: Input CSV file path, containing the feature data to be predicted
- `--output`: Output CSV file path, defaults to predictions.csv

## Input Data Format

The input CSV file should contain feature data, with each row representing a sample and each column representing a feature.

## Output Result Format

The output CSV file contains prediction results, with each row corresponding to a sample in the input data, and each column representing a predicted property value. 