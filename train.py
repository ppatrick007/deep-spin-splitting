#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time

# import the same model definition as predict.py
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

        # 动态计算卷积后的输出维度
        def get_conv_output(input_len, kernel, stride, dilation=1):
            return (input_len - dilation * (kernel - 1) - 1) // stride + 1

        # 计算第一个分支的输出长度
        conv1_1_out = get_conv_output(input_length, 4, 4)
        conv1_2_out = get_conv_output(conv1_1_out, 5, 4)
        self.flatten1_dim = 24 * conv1_2_out

        # 计算第二个分支的输出长度
        conv2_1_out = get_conv_output(input_length, 5, 1, dilation=4)
        conv2_2_out = get_conv_output(conv2_1_out, 4, 4)
        self.flatten2_dim = 24 * conv2_2_out

        # 定义全连接层
        self.fc1 = nn.Linear(self.flatten1_dim + self.flatten2_dim + input_length, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, output_length)

        self.is_inference = False

    def forward(self, x):
        # 标准化
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
        output = act_fn(output)  # apply activation function to the output layer

        if self.is_inference:
            # reverse normalization and exponential transformation in the inference stage
            output = (output - self.Y_pingyi) / self.Y_fangda
            output = (output * self.Y_std) + self.Y_mean
            output = torch.exp(output)
        
        return output

    def set_inference(self, is_inference=True):
        self.is_inference = is_inference
    
    def reset_params(self, X_mean=None, X_std=None, Y_mean=None, Y_std=None, Y_fangda=None, Y_pingyi=None):
        """reset the parameters of the model to custom values. If a value is None, the original value will be kept."""
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
    
    def freeze_conv(self, freeze=True):
        """control whether the convolution layer parameters are trainable"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                for param in module.parameters():
                    param.requires_grad = not freeze
                
        # set eval/train mode
        if freeze:
            self.conv1_1.eval()
            self.conv1_2.eval()
            self.conv2_1.eval()
            self.conv2_2.eval()

def load_model(model_path, input_length=20):
    """load the PyTorch model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file '{model_path}' not found")
    
    print(f"loading model: {model_path}")
    
    # determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the model state dictionary and additional statistics
    checkpoint = torch.load(model_path, map_location=device)
    
    # create a model instance, passing in the loaded mean and standard deviation
    model_class_str = checkpoint['model_class_str']
    if model_class_str == 'ModelN2n':
        model = ModelN2n(
            input_length=input_length,  # the dimension of the input features
            output_length=8,  # the dimension of the output
            X_mean=checkpoint['X_mean'],
            X_std=checkpoint['X_std'],
            Y_mean=checkpoint['Y_mean'],
            Y_std=checkpoint['Y_std'],
            Y_fangda=checkpoint['Y_fangda'],
            Y_pingyi=checkpoint['Y_pingyi']
        )

        # load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("model loaded successfully")
        return model, device
    else:
        raise ValueError(f"unsupported model type: {model_class_str}")

def load_data(x_path, y_path):
    """load the CSV formatted input and label data"""
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"input file '{x_path}' not found")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"label file '{y_path}' not found")
    
    print(f"loading input data: {x_path}")
    print(f"loading label data: {y_path}")
    
    X = pd.read_csv(x_path, header=None).values.astype(np.float32)
    Y = pd.read_csv(y_path, header=None).values.astype(np.float32)
    
    print(f"data loaded successfully, input feature shape: {X.shape}, label shape: {Y.shape}")
    return X, Y

def preprocess_data(X, Y, test_size=0.2, val_size=0.1, random_state=42):
    """preprocess the data and split it into training set, validation set and test set"""
    # take the logarithm of Y
    Y_log = np.log(Y)
    
    # split into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_log, test_size=test_size, random_state=random_state
    )
    
    # split into training set and validation set
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=val_ratio, random_state=random_state
        )
    else:
        X_val, Y_val = X_train[:10], Y_train[:10]  # use a small number of training samples as validation set
    
    print(f"data preprocessed successfully - training set: {X_train.shape[0]} samples, validation set: {X_val.shape[0]} samples, test set: {X_test.shape[0]} samples")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def fine_tune_model(model, X_train, Y_train, X_val, Y_val, output_model_path, epochs=50, batch_size=32, lr=0.0001, freeze_conv=True):
    """fine-tune the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # move the model to GPU (if available)
    model = model.to(device)
    
    # if specified, freeze the convolution layer parameters
    if freeze_conv:
        model.freeze_conv(True)
        print("convolution layer parameters frozen (only train the fully connected layers)")
    
    # convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)
    
    # create data loaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # set the optimizer and loss function
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.L1Loss()  # MAE loss
    
    best_val_loss = float('inf')
    
    # create the directory to save the model (if it doesn't exist)
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    print(f"starting fine-tuning, {epochs} epochs...")
    
    for epoch in range(epochs):
        # train mode
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # validation mode
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'epoch [{epoch+1}/{epochs}], training loss: {avg_train_loss:.6f}, validation loss: {avg_val_loss:.6f}')
        
        # save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': model.X_mean,
                'X_std': model.X_std,
                'Y_mean': model.Y_mean,
                'Y_std': model.Y_std,
                'Y_fangda': model.Y_fangda,
                'Y_pingyi': model.Y_pingyi,
                'model_class_str': 'ModelN2n'
            }, output_model_path)
            print(f"new best model saved to: {output_model_path}")
    
    print(f"fine-tuning completed! best validation loss: {best_val_loss:.6f}")
    return output_model_path

def evaluate_model(model, X_test, Y_test):
    """evaluate the performance of the model on the test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)
    
    # set to inference mode (apply exponential transformation)
    model.set_inference(False)  # evaluate in log space
    model.eval()
    
    # predict
    with torch.no_grad():
        Y_pred = model(X_test_tensor).cpu().numpy()
    
    # calculate MAE (in log space)
    mae = np.mean(np.abs(Y_pred - Y_test))
    
    print(f"测试集MAE (log空间): {mae:.6f}")
    
    # apply exponential transformation, calculate MAE in original space
    Y_pred_orig = np.exp(Y_pred)
    Y_test_orig = np.exp(Y_test)
    mae_orig = np.mean(np.abs(Y_pred_orig - Y_test_orig))
    
    print(f"test set MAE (original space): {mae_orig:.6f}")
    
    return mae, mae_orig

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune model to adapt to new data')
    parser.add_argument('--model', type=str, required=True, help='base model file path (.pth)')
    parser.add_argument('--x_path', type=str, required=True, help='input feature CSV file path')
    parser.add_argument('--y_path', type=str, required=True, help='label CSV file path')
    parser.add_argument('--output_model', type=str, default='fine_tuned_model.pth', help='output model file path')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--freeze_conv', action='store_true', help='freeze the convolution layer')
    parser.add_argument('--model_type', type=str, default='default', 
                        choices=['default', 'PbI', 'PbBr', 'PbCl', 'SnI'], 
                        help='model type (default uses best_model.pth in current directory)')
    return parser.parse_args()

def main():
    """main function"""
    # parse command line arguments
    args = parse_args()
    
    try:
        # determine which model to use based on model_type
        model_path = args.model
        if args.model_type != 'default':
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            model_type_map = {
                'PbI': 'modelPbI',
                'PbBr': 'modelPbBr',
                'PbCl': 'modelPbCl',
                'SnI': 'modelSnI'
            }
            if args.model_type in model_type_map:
                model_subdir = model_type_map[args.model_type]
                # find the best_model file in the directory
                model_files = [f for f in os.listdir(os.path.join(model_dir, model_subdir)) if f.startswith('best_model') and f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(model_dir, model_subdir, model_files[0])
                    print(f"using {args.model_type} model: {model_path}")
                else:
                    print(f"no model file found for type {args.model_type}, using default model.")
        
        # load model
        model, device = load_model(model_path)
        
        # load data
        X, Y = load_data(args.x_path, args.y_path)
        
        # preprocess and split data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(X, Y)
        
        # fine-tune model
        output_model_path = fine_tune_model(
            model, X_train, Y_train, X_val, Y_val, 
            args.output_model, args.epochs, args.batch_size, 
            args.learning_rate, args.freeze_conv
        )
        
        # load fine-tuned model and evaluate
        fine_tuned_model, _ = load_model(output_model_path)
        evaluate_model(fine_tuned_model, X_test, Y_test)
        
        print(f"fine-tuning completed! model saved to: {args.output_model}")
        
    except Exception as e:
        print(f"error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 