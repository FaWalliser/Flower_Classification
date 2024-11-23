import os
import random
import datetime
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
import scipy.io

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
def load_dataset(data_dir):
    img_paths = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(os.listdir(data_dir))}
    
    # Collect images and labels
    for label, idx in label_map.items():
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            img_paths.append(os.path.join(label_dir, img_file))
            labels.append(idx)
    
    # Shuffle dataset
    combined = list(zip(img_paths, labels))
    random.shuffle(combined)
    img_paths, labels = zip(*combined)
    
    return list(img_paths), list(labels), label_map

# 3. Prepare and split dataset
def load_and_prepare_data(data_dir, batch_size=8, test_size=0.2):
    print("Loading dataset...")
    img_paths, labels, label_map = load_dataset(data_dir)
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

    return train_loader, test_loader, label_map

# 4. Training loop
def train_model(train_loader, model, optimizer, criterion, num_epochs=3, device='cpu'):
    model.train()
    scaler = GradScaler()
    image_count = 0  # Counter to track the number of images processed
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader, 1):  # Enumerate to get batch index
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Increment the total image count
            image_count += len(images)

            # Print every 10 images
            if image_count % 10 == 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{current_time} - Processed {image_count} images")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


# 5. Model evaluation function
def evaluate_model(test_loader, model, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

# 6. Main execution function with customizable parameters
def main(data_dir, lr=1e-5, batch_size=8, num_epochs=3, test_size=0.2):
    # Load dataset
    train_loader, test_loader, label_map = load_and_prepare_data(data_dir, batch_size=batch_size, test_size=test_size)

    print("Initializing model...")
    # Load and customize the model
    model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True, num_classes=len(label_map))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Training model...")
    train_model(train_loader, model, optimizer, criterion, num_epochs=num_epochs, device=device)

    # Evaluate the model
    print("Evaluating model...")
    accuracy_test = evaluate_model(test_loader, model, device=device)

    print(f"Final Test Accuracy: {accuracy_test:.2f}%")

# Argument parser for CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer using timm on a flower dataset")
    parser.add_argument("--data_dir", type=str, default="flower_photos", help="Path to the dataset directory")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training (default: 1e-5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training (default: 3)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Percentages of images for the test set")
    
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(data_dir=args.data_dir, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, test_size=args.test_size)
