import scipy.io
import os
import random
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

# 1. Dataset-class
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

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# 2.2. Other Data set
def load_dataset_with_mat(data_dir, mat_file_path):
    # Load labels from .mat-file
    labels_mat = scipy.io.loadmat(mat_file_path)
    num_label = labels_mat['labels'][0]  # Extract labels from array

    img_paths = []
    labels = []

    # Loop over files and assign labels
    for idx, img_file in enumerate(sorted(os.listdir(data_dir))):  # Sorts for correct assignment
        img_paths.append(os.path.join(data_dir, img_file))
        labels.append(num_label[idx] - 1)

    # Shuffle Dataset
    combined = list(zip(img_paths, labels))
    random.shuffle(combined)
    img_paths, labels = zip(*combined)

    # Erstelle die Label-Map aus der .mat-Datei
    unique_labels = set(num_label)  # z. B. {1, 2, ..., 102}
    label_map = {label - 1: f"Class_{label}" for label in unique_labels}

    return list(img_paths), list(labels), label_map

# 3. Prepare and split dataset
def load_and_prepare_data(data_dir, mat_file_path, batch_size=8, test_size=0.2, transform=None):
    img_paths, labels, label_map = load_dataset_with_mat(data_dir, mat_file_path)
    print("Dataset loaded")

    # Split train- and testset
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        img_paths, 
        labels, 
        test_size=test_size, 
        random_state=42, 
        stratify=labels  # Stratified Split - split images according to labels
    )

    print(f"Number of images in training set: {len(train_paths)}")
    print(f"Number of images in test set: {len(test_paths)}")

    # Image transformation using `create_transform`
    transform = create_transform(
        input_size=224,                # Resize all images to 224x224
        is_training=True,              # Enable training-specific augmentations (e.g., random crop, flip)
        mean=(0.5, 0.5, 0.5),          # Normalization mean for RGB channels
        std=(0.5, 0.5, 0.5),           # Normalization standard deviation for RGB channels
        color_jitter=0.2,              # Apply color jitter with brightness, contrast, saturation, and hue
        re_prob=0,                     # Disable random erasing
        re_mode=None,                  # No erasing mode used
        re_count=0,                    # No erasing operations
        interpolation='bilinear'       # Use bilinear interpolation for resizing
    )


    # Create Dataset and DataLoader
    train_dataset = FlowerDataset(train_paths, train_labels, transform=transform)
    test_dataset = FlowerDataset(test_paths, test_labels, transform=transform)

    
    # Data Augmentation for training
    # transform_train = create_transform(
    #     input_size=224,                # Resize images to 224x224
    #     is_training=True,              # Enable training-specific augmentations (e.g., random crop, flip)
    #     mean=(0.5, 0.5, 0.5),          # Normalization mean for RGB channels
    #     std=(0.5, 0.5, 0.5),           # Normalization standard deviation for RGB channels
    #     color_jitter=0.2,              # Apply color jitter for brightness, contrast, saturation, and hue
    #     interpolation='bilinear'       # Use bilinear interpolation for resizing
    # )

    # Transformation for testing (no data augmentation)
    # transform_test = create_transform(
    #     input_size=224,                # Resize images to 224x224
    #     is_training=False,             # Disable training-specific augmentations
    #     mean=(0.5, 0.5, 0.5),          # Normalization mean for RGB channels
    #     std=(0.5, 0.5, 0.5),           # Normalization standard deviation for RGB channels
    #     interpolation='bilinear'       # Use bilinear interpolation for resizing
    # )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_map

# 4. Training loop
def train_model(train_loader, model, optimizer, criterion, num_epochs=3, device='cpu'):
    model.train()  # Set the model to training mode
    scaler = GradScaler()
    total_train_correct = 0
    total_train_images = 0
    init_t = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        image_count = 0
        # Gesamte Bilderanzahl im Trainingsset für die Epoche
        epoch_images = len(train_loader.dataset)

        for images, labels in train_loader:
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
            
            # Compute correct predictions and total count
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)  # Increment the total count

            # Increment the total image count
            image_count += len(images)

            # Print number and estimated time
            if image_count % (100 * len(images)) == 0:  # Print every 10 Batches
                elapsed_time = time.time() - init_t
                total_images = epoch_images * num_epochs
                
                # Geschätzte verbleibende Zeit
                time_per_image = (time.time() - epoch_start_time) / image_count
                estimated_remaining_time = time_per_image * (total_images - (image_count + epoch * epoch_images))
                
                print(f"Processed {image_count}/{epoch_images} images, "
                  f"Elapsed time: {elapsed_time:.2f} seconds, "
                  f"Estimated remaining time: {estimated_remaining_time:.2f} seconds")
        
        # Compute accuracy for this epoch
        epoch_train_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, Training Accuracy: {epoch_train_accuracy:.2f}%")
        
        # Update total correct predictions and total images
        total_train_correct += correct
        total_train_images += total
    
    # Analyze training accuracy
    total_train_accuracy = 100 * total_train_correct / total_train_images
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, "
          f"Training Accuracy: {epoch_train_accuracy:.2f}%, "
          f"Time taken: {time.time() - epoch_start_time:.2f} seconds")

    # Analyze training time
    end_t = time.time()
    total_time = end_t - init_t
    print(f"Model training time: {total_time:.2f} seconds")
    return total_train_accuracy, epoch_train_accuracy, total_time, average_loss

# 5. Model evaluation function
def evaluate_model(test_loader, model, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    processed_images = 0
    init_test_t = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Ensure labels are of type torch.long
            labels = labels.long()  # Cast labels to long (int64)

            # Then compute the loss
            outputs = model(images)

            loss = criterion(outputs, labels)  # Calculate loss for the batch

            # Accumulate the loss
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            processed_images += labels.size(0)  # Zähle die verarbeiteten Bilder
        
            # Print number and estimated time
            if processed_images % 100 == 0:  # Print every 10 Batches
                elapsed_time = time.time() - init_test_t  # Elapsed time
                total_images = len(test_loader.dataset)
                remaining_images = total_images - processed_images
                
                time_per_image = elapsed_time / processed_images
                estimated_remaining_time = time_per_image * remaining_images
                
                # Ausgabe der verstrichenen Zeit und verbleibenden Zeit
                print(f"Image tested number {processed_images}/{total_images}, "
                      f"Elapsed time: {elapsed_time:.2f}, "
                      f"Estimated remaining time: {estimated_remaining_time:.2f}")

    end_test_t = time.time()
    total_time = end_test_t - init_test_t
    print(f"Model testing time: {total_time} seconds")

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # After loop, calculate average loss
    average_loss = total_loss / len(test_loader)
    return accuracy, total_time, average_loss

# 6. Main execution function with customizable parameters
def main(data_dir, mat_file_path, lr=1e-5, batch_size=8, num_epochs=3, test_size=0.2, dropout=0.0):
    # Load dataset
    train_loader, test_loader, label_map = load_and_prepare_data(data_dir, mat_file_path, batch_size=batch_size, test_size=test_size)

    print("Train model:")
    # Load and customize the model
    model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=102)  # Set num_classes=102 for Flowers-102

    # Überprüfe die aktuelle Struktur des Heads
    #print("Model:")
    #print(model)

    # Manipulate dropout rate
    if(dropout > 0):
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),  # Set dropout rate (0.3)
            model.head
        )

    # Optimizer and Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01) #lr: learnign rate: 0.00001
    criterion = nn.CrossEntropyLoss()

    # Train the model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    accuracy_training, accuracy_last_epoch, time_training, train_loss = train_model(train_loader, model, optimizer, criterion, num_epochs=num_epochs, device=device)

    # Evaluate the model
    accuracy_test, time_evaluation, test_loss = evaluate_model(test_loader, model, device=device)

    return accuracy_training, accuracy_test, accuracy_last_epoch, time_training, time_evaluation, train_loss, test_loss

# Argumentparser für Kommandozeilenparameter
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on a flower dataset")
    parser.add_argument("--data_dir", type=str, default="folwers-102-categories-perso/jpg", help="Path to the dataset directory")
    parser.add_argument("--mat_file_dir", type=str, default="folwers-102-categories-perso/imagelabels.mat", help="Path to the .mat file")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training (default: 1e-5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training (default: 3)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Percentages of images for the test set")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for training (default: 0.0)")
    parser.add_argument("--output_file", type=str, default="results.txt", help="Path to the results file")
    
    args = parser.parse_args()

    # Hauptprogramm mit den Argumenten starten
    accuracy_training, accuracy_test, accuracy_last_epoch, time_training, time_evaluation, train_loss, test_loss = main(data_dir=args.data_dir, mat_file_path=args.mat_file_dir, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, test_size=args.test_size, dropout=args.dropout)

     # Ausgabe der Ergebnisse
    print(f"Accuracy Train Set: {accuracy_training:.2f}%")
    print(f"Accuracy Train Set (Last Epoch): {accuracy_last_epoch:.2f}%")
    print(f"Accuracy Test Set: {accuracy_test:.2f}%")
    print(f"Training Time: {time_training:.2f}s")
    print(f"Evaluation Time: {time_evaluation:.2f}s")
    print(f"Training Loss: {train_loss:.2f}s")
    print(f"Evaluation Loss: {test_loss:.2f}s")

    # Optional: Speichere die Ergebnisse in eine Datei
    if args.output_file:
        with open(args.output_file, "a") as f:
            f.write(f"LR: {args.lr}, Batch Size: {args.batch_size}, Dropout: {args.dropout}, "
                    f"Accuracy Train Set: {accuracy_training:.4f}%, "
                    f"Accuracy Train Set (Last Epoch): {accuracy_last_epoch:.4f}%, "
                    f"Accuracy Test Set: {accuracy_test:.4f}%, "
                    f"Training Time: {time_training:.2f}s, "
                    f"Evaluation Time: {time_evaluation:.2f}s, "
                    f"Training Loss: {train_loss:.4f}s, "
                    f"Evaluation Loss: {test_loss:.4f}s\n")