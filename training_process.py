#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Training script for VUDS-AI: train one model per label condition
import argparse
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Training script for VUDS-AI: train one model per label condition")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to processed dataset directory containing metadata/")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["pftg", "pfus", "xray"],
                        help="Dataset type: pftg, pfus, or xray")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "densenet121"],
                        help="Backbone model architecture")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs per condition")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save trained models")
    return parser.parse_args()


class ImageDataset(Dataset):
    # Custom dataset class for loading images and labels
    def __init__(self, images_dir, labels_df, transform=None):
        self.images_dir = images_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return image, label


def build_model(backbone_name, num_classes):
    # Build the backbone model and adjust final layer
    if backbone_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    return model


def train_condition(data_dir, model_type, condition, backbone, epochs, batch_size, lr, output_dir):
    # Train a model for a single condition label
    csv_path = os.path.join(data_dir, 'metadata', f"{model_type}_train.csv")
    df = pd.read_csv(csv_path)
    if condition not in df.columns:
        raise KeyError(f"Condition '{condition}' not found in CSV columns")

    # Handle NaN values and comma-separated labels
    cond_df = df[['filename', condition]].rename(columns={condition: 'label'})
    
    # Convert labels to numeric values
    def convert_label(x):
        if pd.isna(x):
            return 0
        try:
            # If it's a comma-separated string, take the first value
            if isinstance(x, str) and ',' in x:
                return int(x.split(',')[0])
            return int(float(x))
        except (ValueError, TypeError):
            return 0
    
    cond_df['label'] = cond_df['label'].apply(convert_label)

    # Print positive/negative distribution
    pos = cond_df['label'].sum()
    neg = len(cond_df) - pos
    print(f"Condition {condition}: Positive={pos}, Negative={neg}")

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images_dir = os.path.join(data_dir, 'images', model_type)
    dataset = ImageDataset(images_dir, cond_df, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    num_classes = cond_df['label'].nunique()
    model = build_model(backbone, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[{condition}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save trained model
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_type}_{condition}_{backbone}.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved model for {condition} to {out_path}\n")


def prepare_dataset(data_dir, model_type):
    # Prepare training CSV from metadata
    metadata_path = os.path.join(data_dir, 'metadata', f'{model_type}_samples.csv')
    df = pd.read_csv(metadata_path)
    print("DEBUG: Loaded metadata file ({}) head:\n{}".format(metadata_path, df.head()))

    # Filter only .jpg files
    df = df[df['img_path'].str.endswith('.jpg')].reset_index(drop=True)

    # Create training DataFrame
    train_df = pd.DataFrame()
    train_df['filename'] = df['img_path'].apply(lambda x: x.split('/')[-1])

    # Get all label columns (excluding metadata columns)
    label_columns = [col for col in df.columns if col not in ['sample_id', 'patient_id', 'report_id', 'img_path']]
    
    # Copy labels directly from original DataFrame
    for label in label_columns:
        train_df[label] = df[label]

    # Save processed dataset
    output_path = os.path.join(data_dir, 'metadata', f'{model_type}_train.csv')
    train_df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

    # Print label distribution summary
    print("\nLabel distribution:")
    for label in label_columns:
        positive = (train_df[label] == 1).sum()
        negative = (train_df[label] == 0).sum()
        print(f"{label}: Positive={positive}, Negative={negative}")

    return output_path, len(label_columns)


# Mapping from label to dataset type
LABEL_TO_DATASET = {
    "Detrusor_instability": "pftg",
    "Flow_pattern": "pfus",
    "EMG_ES_relaxation": "pfus",
    "Trabeculation": "xray",
    "Diverticulum": "xray",
    "Cystocele": "xray",
    "VUR": "xray",
    "Bladder_neck_relaxation": "xray",
    "External_sphincter_relaxation": "xray",
    "Pelvic_floor_relaxation": "xray"
}


def main():
    args = parse_args()
    # Generate training CSV and get number of conditions
    csv_path, num_conditions = prepare_dataset(args.data_dir, args.model_type)
    df = pd.read_csv(csv_path)

    # Select conditions relevant to the model_type
    conditions = [c for c in df.columns if c != 'filename' and LABEL_TO_DATASET.get(c) == args.model_type]
    print(f"Loaded training CSV: {csv_path}")

    # Train a model for each condition
    for condition in conditions:
        train_condition(
            args.data_dir,
            args.model_type,
            condition,
            args.backbone,
            args.epochs,
            args.batch_size,
            args.lr,
            args.output_dir
        )


if __name__ == '__main__':
    main()