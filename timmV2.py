import os
import random
import datetime
import scipy.io
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import create_transform
from torch.cuda.amp import GradScaler, autocast
import argparse

# 1. Dataset class for flower_photos
class FlowerDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # Open image and apply transformations
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. Function to load dataset
def load_dataset(data_dir, labels_file):
    # Load the labels from the .mat file
    labels_mat = scipy.io.loadmat(labels_file)
    labels = labels_mat['labels'][0]  # The labels array is stored in 'labels'

    # Ensure labels are zero-indexed
    labels = [label - 1 for label in labels]  # Subtract 1 from each label

    # Get image paths from the flowers-102-categories-perso/jpg folder
    img_paths = [os.path.join(data_dir, img_file) for img_file in os.listdir(data_dir) if img_file.endswith('.jpg')]

    # Check if the number of images and labels match
    assert len(img_paths) == len(labels), f"Number of images: {len(img_paths)} does not match number of labels: {len(labels)}"

    return img_paths, labels

# 3. Prepare and split dataset
def load_and_prepare_data(data_dir, labels_file, batch_size=8, test_size=0.2):
    print("Loading dataset...")
    img_paths, labels = load_dataset(data_dir, labels_file)  # No label map needed anymore
    print("Dataset loaded")

    # Split train- and testset
    train_paths, test_paths, train_labels, test_labels = train_test_split(img_paths, labels, test_size=test_size, random_state=42)

    print(f"Number of images in training set: {len(train_paths)}")
    print(f"Number of images in test set: {len(test_paths)}")

    # Image transformation
    transform = create_transform(
        input_size=224,
        is_training=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )

    # Create Dataset and DataLoader
    train_dataset = FlowerDataset(train_paths, train_labels, transform=transform)
    test_dataset = FlowerDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 4. Training loop
def train_model(train_loader, model, optimizer, criterion, num_epochs=3, device='cpu'):
    model.train()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        image_count = 0  # Counter to track the number of images processed
        for i, (images, labels) in enumerate(train_loader, 1):  # Enumerate to get batch index
            images, labels = images.to(device), labels.to(device)
            
            # Convert labels to long type (torch.long)
            labels = labels.long()

            optimizer.zero_grad()

            # Ensure that mixed precision is handled correctly with autocast
            with autocast():  # Mixed precision
                # Forward pass: Pass images to the model to get the outputs
                outputs = model(images)
                
                # Ensure outputs is not None or empty
                if outputs is None or outputs.size(0) != images.size(0):
                    print(f"Error: Outputs shape mismatch. Expected batch size {images.size(0)}, but got {outputs.size(0) if outputs is not None else 'None'}")
                    continue  # Skip this batch if outputs are invalid

                # Calculate loss
                loss = criterion(outputs, labels)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Increment the total image count
            image_count += len(images)

            # Print every 1000 images
            if image_count % 1000 == 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{current_time} - Processed {image_count} images in current Epoch")

        accuracy_training = 100 * correct / total
        # Log stats for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    return accuracy_training


# 5. Model evaluation function
def evaluate_model(test_loader, model, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Ensure labels are of type torch.long
            labels = labels.long()  # Cast labels to long (int64)

            # Then compute the loss
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_test = 100 * correct / total
    time_evaluation = time.time() - start_time
    print(f"Accuracy on test set: {accuracy_test:.2f}%")
    return accuracy_test, time_evaluation

# 6. Main execution function with customizable parameters
def main(data_dir, labels_file, lr=1e-5, batch_size=8, num_epochs=3, test_size=0.2):
    # Load dataset
    train_loader, test_loader = load_and_prepare_data(data_dir, labels_file, batch_size=batch_size, test_size=test_size)

    print("Initializing model...")
    # Load and customize the model
    model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=102)  # Set num_classes=102 for Flowers-102
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Training model...")
    start_time = time.time()
    accuracy_training = train_model(train_loader, model, optimizer, criterion, num_epochs=num_epochs, device=device)
    time_training = time.time() - start_time

    # Evaluate the model
    print("Evaluating model...")
    accuracy_test, time_evaluation = evaluate_model(test_loader, model, device=device)

    # Save results to file
    with open('training_results.txt', 'a') as f:
        f.write(f"LR: {lr}, Batch Size: {batch_size}, "
                f"Accuracy Train Set: {accuracy_training:.2f}%, "
                f"Accuracy Test Set: {accuracy_test:.2f}%, "
                f"Training Time: {time_training:.2f}s, "
                f"Evaluation Time: {time_evaluation:.2f}s\n")

    print(f"Final Test Accuracy: {accuracy_test:.2f}%")

# Argument parser for CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer using timm on a flower dataset")
    parser.add_argument("--data_dir", type=str, default="flowers-102-categories-perso/jpg", help="Path to the image directory")
    parser.add_argument("--labels_file", type=str, default="flowers-102-categories-perso/imagelabels.mat", help="Path to the labels .mat file")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training (default: 1e-5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training (default: 3)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Percentages of images for the test set")
    
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(data_dir=args.data_dir, labels_file=args.labels_file, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, test_size=args.test_size)
