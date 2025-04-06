#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from geometry_x import geometry_to_X

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch process geometry.in files and generate predictions')
    parser.add_argument('--model', type=str, required=True, help='PyTorch model file path (.pth)')
    parser.add_argument('--geometrydir', type=str, required=True, help='Directory containing geometry.in files')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file path')
    parser.add_argument('--csvoutput', type=str, default='X.csv', help='Feature output CSV file path')
    return parser.parse_args()

def find_geometry_files(directory):
    """Recursively find all geometry.in files"""
    geometry_files = []
    
    # Only search in subdirectories, not the root directory itself
    # This prevents duplicate counting
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'geometry.in':
                geometry_files.append(os.path.join(root, file))
    
    return geometry_files

def process_geometry_files(geometry_files):
    """Process all geometry.in files and generate feature vectors"""
    features = []
    file_paths = []
    
    for file_path in geometry_files:
        try:
            # Get the directory path of the file
            file_dir = os.path.dirname(file_path)
            file_name = 'geometry.in'
            
            # Process geometry.in file to generate feature vector
            feature = geometry_to_X(file=file_name, path=file_dir, angle='rad', dimension=3)
            features.append(feature)
            file_paths.append(file_path)
            print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return np.array(features), file_paths

def save_features_to_csv(features, file_paths, output_path):
    """Save feature vectors to CSV file"""
    df = pd.DataFrame(features)
    
    # Save to CSV file without index or header
    df.to_csv(output_path, header=False, index=False)
    print(f"Features saved to: {output_path}")
    
    return output_path

def run_prediction(model_path, input_path, output_path):
    """Run prediction"""
    from predict import load_model, load_data, preprocess_data, make_predictions, save_predictions
    
    try:
        # Load model
        model, device = load_model(model_path)
        
        # Load and preprocess data
        data = load_data(input_path)
        X = preprocess_data(data)
        
        # Make predictions
        predictions = make_predictions(model, X, device)
        
        # Save prediction results
        save_predictions(predictions, data, output_path)
        
        print(f"Prediction completed, results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return 1
    
    return 0

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Find all geometry.in files
        print(f"Finding geometry.in files...")
        geometry_files = find_geometry_files(args.geometrydir)
        print(f"Found {len(geometry_files)} geometry.in files")
        
        if not geometry_files:
            print(f"Error: No geometry.in files found in {args.geometrydir}")
            return 1
        
        # Process all geometry.in files
        print("Processing geometry.in files...")
        features, file_paths = process_geometry_files(geometry_files)
        
        # Save feature vectors to CSV file
        csv_path = save_features_to_csv(features, file_paths, args.csvoutput)
        
        # Run prediction
        print("Running prediction...")
        result = run_prediction(args.model, csv_path, args.output)
        
        if result == 0:
            print(f"Batch processing completed! Prediction results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 