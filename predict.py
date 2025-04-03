#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelN2n(nn.Module):
    def __init__(self, input_length, output_length, X_mean=None, X_std=None, Y_mean=None, Y_std=None, Y_fangda=0.6, Y_pingyi=1, activation='elu'):
        super(ModelN2n, self).__init__()

        self.activation = activation
        
        self.X_mean = X_mean if X_mean is not None else torch.zeros(input_length)
        self.X_std = X_std if X_std is not None else torch.ones(input_length)

        self.Y_mean = Y_mean if Y_mean is not None else torch.zeros(output_length)
        self.Y_std = Y_std if Y_std is not None else torch.ones(output_length)

        self.Y_fangda = Y_fangda if Y_fangda is not None else 1.0
        self.Y_pingyi = Y_pingyi if Y_pingyi is not None else 0.0

        self.X_mean.requires_grad = False
        self.X_std.requires_grad = False
        self.Y_mean.requires_grad = False
        self.Y_std.requires_grad = False

        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=4, stride=4)
        self.conv1_2 = nn.Conv1d(32, 24, kernel_size=5, stride=4)
        self.conv2_1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, dilation=4)
        self.conv2_2 = nn.Conv1d(32, 24, kernel_size=4, stride=4)

        # dynamically calculate output dimension after convolution
        def get_conv_output(input_len, kernel, stride, dilation=1):
            return (input_len - dilation * (kernel - 1) - 1) // stride + 1

        # calculate output length of first branch
        conv1_1_out = get_conv_output(input_length, 4, 4)
        conv1_2_out = get_conv_output(conv1_1_out, 5, 4)
        self.flatten1_dim = 24 * conv1_2_out

        # calculate output length of second branch
        conv2_1_out = get_conv_output(input_length, 5, 1, dilation=4)
        conv2_2_out = get_conv_output(conv2_1_out, 4, 4)
        self.flatten2_dim = 24 * conv2_2_out

        # define fully connected layers
        self.fc1 = nn.Linear(self.flatten1_dim + self.flatten2_dim + input_length, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, output_length)

        self.is_inference = False

    def forward(self, x):
        # normalize
        x = (x - self.X_mean) / self.X_std
        act_fn = getattr(F, self.activation)

        # process branch 1
        x1 = x.unsqueeze(1)  # [batch, 1, input_len]
        x1 = act_fn(self.conv1_1(x1))
        x1 = act_fn(self.conv1_2(x1))
        x1 = torch.flatten(x1, 1)  # [batch, 24 * conv1_2_out]

        # process branch 2
        x2 = x.unsqueeze(1)  # [batch, 1, input_len]
        x2 = act_fn(self.conv2_1(x2))
        x2 = act_fn(self.conv2_2(x2))
        x2 = torch.flatten(x2, 1)  # [batch, 24 * conv2_2_out]

        # concatenate features
        z = torch.cat([x1, x2, x], dim=1)  # [batch, flatten1_dim + flatten2_dim + input_length]

        # fully connected layers
        z = act_fn(self.fc1(z))
        z = act_fn(self.fc2(z))
        z = act_fn(self.fc3(z))
        z = act_fn(self.fc4(z))
        output = self.output(z)
        output = act_fn(output)  # apply activation function to output layer, consistent with Keras

        if self.is_inference:
            # reverse normalization and exponential transformation in inference stage
            output = (output - self.Y_pingyi) / self.Y_fangda
            output = (output * self.Y_std) + self.Y_mean
            output = torch.exp(output)
        
        return output

    def set_inference(self, is_inference=True):
        self.is_inference = is_inference
    
    def reset_params(self, X_mean=None, X_std=None, Y_mean=None, Y_std=None, Y_fangda=None, Y_pingyi=None):
        """
        reset model parameters to custom values. if a value is None, keep the original value unchanged.
        """
        if X_mean is not None:
            self.X_mean = X_mean
        if X_std is not None:
            self.X_std = X_std
        if Y_mean is not None:
            self.Y_mean = Y_mean
        if Y_std is not None:
            self.Y_std = Y_std
        if Y_fangda is not None:
            self.Y_fangda = Y_fangda
        if Y_pingyi is not None:
            self.Y_pingyi = Y_pingyi

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='deep learning model for prediction')
    parser.add_argument('--model', type=str, required=True, help='PyTorch model file path(.pth)')
    parser.add_argument('--input', type=str, required=True, help='input CSV file path')
    parser.add_argument('--output', type=str, default='predictions.csv', help='output CSV file path')
    return parser.parse_args()

def load_model(model_path, input_length=20):
    """load PyTorch model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file '{model_path}' not found")
    
    print(f"loading model: {model_path}")
    
    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model state dictionary and additional statistics
    checkpoint = torch.load(model_path, map_location=device)
    
    # create model instance, pass loaded mean and standard deviation
    model_class_str = checkpoint['model_class_str']
    if model_class_str == 'ModelN2n':
        model = ModelN2n(
            input_length=input_length,  # 输入特征的维度
            output_length=8,  # 输出的维度
            X_mean=checkpoint['X_mean'],
            X_std=checkpoint['X_std'],
            Y_mean=checkpoint['Y_mean'],
            Y_std=checkpoint['Y_std'],
            Y_fangda=checkpoint['Y_fangda'],
            Y_pingyi=checkpoint['Y_pingyi']
        )

        # load model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # switch to inference mode
        model.set_inference(True)
        model.eval()
        model.to(device)
        
        print("model loaded successfully")
        return model, device
    else:
        raise ValueError(f"unsupported model type: {model_class_str}")

def load_data(input_path):
    """load CSV format input data"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input file '{input_path}' not found")
    
    print(f"loading input data: {input_path}")
    data = pd.read_csv(input_path)
    print(f"data loaded successfully, {len(data)} rows")
    return data

def preprocess_data(data):
    """preprocess input data"""
    # assume input data needs to be converted to numpy array, then convert to tensor
    X = data.values.astype(np.float32)
    print(f"data preprocessed successfully, input feature shape: {X.shape}")
    return X

def make_predictions(model, X, device):
    """make predictions using the model"""
    print("starting prediction...")
    
    # convert data to tensor and move to corresponding device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # make predictions
    with torch.no_grad():
        predictions = model(X_tensor)
    
    # convert predictions to numpy array and move to CPU
    predictions = predictions.cpu().numpy()
    
    print("prediction completed")
    return predictions

def save_predictions(predictions, original_data, output_path):
    """save predictions to CSV file"""
    # create a DataFrame for predictions
    pred_columns = [f'property_{i}' for i in range(predictions.shape[1])]
    pred_df = pd.DataFrame(predictions, columns=pred_columns)
    
    # if original data has ID column, add it to the results
    if 'id' in original_data.columns:
        pred_df.insert(0, 'id', original_data['id'])
    
    print(f"saving predictions to: {output_path}")
    pred_df.to_csv(output_path, index=False, header=False)
    print(f"predictions saved")

def main():
    """main function"""
    # parse command line arguments
    args = parse_args()
    
    try:
        # load model
        model, device = load_model(args.model)
        
        # load and preprocess data
        data = load_data(args.input)
        X = preprocess_data(data)
        
        # make predictions
        predictions = make_predictions(model, X, device)
        
        # save predictions
        save_predictions(predictions, data, args.output)
        
        print(f"processing completed! predictions saved to: {args.output}")
        
    except Exception as e:
        print(f"error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 