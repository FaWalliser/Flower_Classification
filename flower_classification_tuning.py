import os
import random
import datetime
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import argparse

# 1. Dataset class for flower_photos
class FlowerDataset(Dataset):
    def __init__(self, img_paths, labels, processor, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # Open image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transform for resizing and normalization
        if self.transform:
            image = self.transform(image)
        else:
            # Use processor for resizing and normalizing
            image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze()
        
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
def load_and_prepare_data(data_dir, batch_size=8, test_size=0.2, transform=None):
    processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224-in21k')
    print("Processor initialized")

    img_paths, labels, label_map = load_dataset(data_dir)
    print("Dataset loaded")

    # Split train- and testset
    train_paths, test_paths, train_labels, test_labels = train_test_split(img_paths, labels, test_size=test_size, random_state=42, stratify=labels)

    print(f"Number of images in training set: {len(train_paths)}")
    print(f"Number of images in test set: {len(test_paths)}")

    # Image transformation (resize and normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #improve colors
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    # Create Dataset and DataLoader
    train_dataset = FlowerDataset(train_paths, train_labels, processor, transform=transform)
    test_dataset = FlowerDataset(test_paths, test_labels, processor, transform=transform)

    # Data Augmentation
    #transform_train = transforms.Compose([
    #    transforms.RandomResizedCrop(224),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #])
    #transform_test = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_map

# 4. Training loop
def train_model(train_loader, model, optimizer, criterion, num_epochs=3, device='cpu'):
    model.train()  # Set the model to training mode
    i = 0
    total_train_correct = 0
    total_train_images = 0
    init_t: datetime = datetime.datetime.now()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Compute correct predictions and total count
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)  # Increment the total count

            i += 1
            if(i % 100 == 2):
                print(f"Image processed number {i}")
        
        # Compute accuracy for this epoch
        epoch_train_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, Training Accuracy: {epoch_train_accuracy:.2f}%")
        
        # Update total correct predictions and total images
        total_train_correct += correct
        total_train_images += total
    
    # Analyze training accuracy
    total_train_accuracy = 100 * total_train_correct / total_train_images
    print(f"Final Training Accuracy: {total_train_accuracy}%")

    # Analyze training time
    end_t: datetime = datetime.datetime.now()
    total_time = end_t - init_t
    print(f"Model training time: {total_time.total_seconds():.2f} seconds")
    return total_train_accuracy, epoch_train_accuracy, total_time

# 5. Model evaluation function
def evaluate_model(test_loader, model, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    j = 0
    init_test_t: datetime = datetime.datetime.now()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(pixel_values=images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            j += 1
            if(j % 100 == 2):
                print(f"Image tested number {j}")

    end_test_t: datetime = datetime.datetime.now()
    total_time = end_test_t - init_test_t
    print(f"Model testing time: {total_time.total_seconds()} seconds")

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy, total_time

# 6. Main execution function with customizable parameters
def main(data_dir, lr=1e-5, batch_size=8, num_epochs=3, test_size=0.2):
    # Load dataset
    train_loader, test_loader, label_map = load_and_prepare_data(data_dir, batch_size=batch_size, test_size=test_size)

    print("Train model:")
    # Load and customize the model
    model = ViTForImageClassification.from_pretrained('vit-base-patch16-224-in21k', num_labels=len(label_map))
    model.train()

    # Optimizer and Loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01) #lr: learnign rate: 0.00001
    criterion = nn.CrossEntropyLoss()

    # Train the model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    accuracy_training, accuracy_last_epoch, time_training = train_model(train_loader, model, optimizer, criterion, num_epochs=num_epochs, device=device)

    # Evaluate the model
    accuracy_test, time_evaluation = evaluate_model(test_loader, model, device=device)

    return accuracy_training, accuracy_test, accuracy_last_epoch, time_training, time_evaluation

# Argumentparser für Kommandozeilenparameter
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on a flower dataset")
    parser.add_argument("--data_dir", type=str, default="flower_photos", help="Path to the dataset directory")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training (default: 1e-5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training (default: 3)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Percentages of images for the test set")
    parser.add_argument("--output_file", type=str, default="results", help="Path to the results file")
    
    args = parser.parse_args()

    # Hauptprogramm mit den Argumenten starten
    accuracy_training, accuracy_test, accuracy_last_epoch, time_training, time_evaluation = main(data_dir=args.data_dir, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, test_size=args.test_size)

     # Ausgabe der Ergebnisse
    print(f"Accuracy Train Set: {accuracy_training:.2f}%")
    print(f"Accuracy Train Set (Last Epoch): {accuracy_last_epoch:.2f}%")
    print(f"Accuracy Test Set: {accuracy_test:.2f}%")
    print(f"Training Time: {time_training.total_seconds():.2f}s")
    print(f"Evaluation Time: {time_evaluation.total_seconds():.2f}s")

    # Optional: Speichere die Ergebnisse in eine Datei
    if args.output_file:
        with open(args.output_file, "a") as f:
            f.write(f"LR: {args.lr}, Batch Size: {args.batch_size}, "
                    f"Accuracy Train Set: {accuracy_training:.2f}%, "
                    f"Accuracy Train Set (Last Epoch): {accuracy_last_epoch:.2f}%, "
                    f"Accuracy Test Set: {accuracy_test:.2f}%, "
                    f"Training Time: {time_training.total_seconds():.2f}s, "
                    f"Evaluation Time: {time_evaluation.total_seconds():.2f}s\n")