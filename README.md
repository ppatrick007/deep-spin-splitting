# HOIPs Deep Learning Spin Splitting Prediction Tool

This is a deep learning model and executable Python script for predicting spin splitting in perovskite materials.

## Installation

First, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Direct Prediction from CSV

Run the prediction using the following command:

```bash
python predict.py --model MODEL_PATH --input INPUT_PATH --output OUTPUT_PATH
```

Example:

```bash
python predict.py --model ./models/modelPbBr/best_model.pth --input ./X.csv --output predictions.csv
```

Here, `X.csv` is our example input file indicating the format we use.

#### Parameters

- `--model`: PyTorch model file path (.pth format)
- `--input`: Input CSV file path, containing the feature data to be predicted
- `--output`: Output CSV file path, defaults to predictions.csv
- `--model_type`: Model type to use, choices are ['default', 'PbI', 'PbBr', 'PbCl', 'SnI'], defaults to 'default'

### Batch Processing from Geometry Files

For processing multiple `geometry.in` files in a directory structure, use the batch prediction script:

```bash
python batch_predict.py --model MODEL_PATH --geometrydir GEOMETRY_DIR_PATH --output OUTPUT_PATH
```

Example:

```bash
python batch_predict.py --model ./models/modelPbBr/best_model.pth --geometrydir ./geometrydata --output predictions.csv
```

This script will:
1. Find all `geometry.in` files within the specified directory and its subdirectories
2. Process each file to extract features
3. Combine features into a single CSV file
4. Run the prediction model on the features
5. Save the prediction results

#### Parameters

- `--model`: PyTorch model file path (.pth format)
- `--geometrydir`: Directory containing geometry.in files (can contain multiple subdirectories)
- `--output`: Output CSV file path, defaults to predictions.csv
- `--csvoutput`: Intermediate feature CSV file path, defaults to X.csv
- `--model_type`: Model type to use, choices are ['default', 'PbI', 'PbBr', 'PbCl', 'SnI'], defaults to 'default'

### Model Fine-tuning

If you have new data (input features and corresponding labels), you can fine-tune an existing model:

```bash
python train.py --model MODEL_PATH --x_path X_PATH --y_path Y_PATH --output_model OUTPUT_MODEL_PATH
```

Example:

```bash
python train.py --model ./models/modelPbBr/best_model.pth --x_path ./new_X.csv --y_path ./new_Y.csv --output_model ./fine_tuned_model.pth
```

#### Parameters

- `--model`: Base model file path (.pth format)
- `--x_path`: Input features CSV file path
- `--y_path`: Labels CSV file path
- `--output_model`: Output model file path, defaults to fine_tuned_model.pth
- `--epochs`: Number of training epochs, defaults to 50
- `--batch_size`: Batch size for training, defaults to 32
- `--learning_rate`: Learning rate for training, defaults to 0.0001
- `--freeze_conv`: Flag to freeze convolutional layers, only train fully connected layers
- `--model_type`: Base model type, choices are ['default', 'PbI', 'PbBr', 'PbCl', 'SnI'], defaults to 'default'

## Input Data Format

The input CSV file should contain feature data, with each row representing a sample and each column representing a feature. It should not include column names or header rows.

## Output Result Format

The output CSV file contains prediction results, with each row corresponding to a sample in the input data, and each column representing a predicted property value. 

Also in the folder `testcase` the user can find related datasets for each model.