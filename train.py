import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import os
from PIL import Image

# =============================================================================
# Part 1: CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Dataset/Train"
VAL_DIR = "Dataset/Test"
MODEL_SAVE_PATH = "saved_models/best_leaf_disease_model.pth"

# Model Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20
IMG_SIZE = 224

# Data Augmentation
TRAIN_TRANSFORMS_CONFIG = {
    'horizontal_flip_p': 0.5,
    'rotation_degrees': 10,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
}
VAL_TRANSFORMS_CONFIG = {
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
}


# =============================================================================
# Part 2: MODEL DEFINITION
# =============================================================================
def build_model(num_classes, pretrained=True, fine_tune=True):
    """Builds the ResNet-50 model."""
    print('Building model...')
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print('Model built successfully!')
    return model


# =============================================================================
# Part 3: DATASET AND DATALOADERS
# =============================================================================
class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root_dir, class_name)
            
            for subdir, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(subdir, fname)
                        item = (path, class_idx)
                        self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = Image.open(path).convert("RGB")
        if self.transform:
            sample = self.transform(sample)
        return sample, target

def get_train_transforms():
    """Returns transformations for the training dataset."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=TRAIN_TRANSFORMS_CONFIG['horizontal_flip_p']),
        transforms.RandomRotation(TRAIN_TRANSFORMS_CONFIG['rotation_degrees']),
        transforms.ToTensor(),
        # --- THIS LINE IS NOW FIXED ---
        transforms.Normalize(mean=TRAIN_TRANSFORMS_CONFIG['normalize_mean'], std=TRAIN_TRANSFORMS_CONFIG['normalize_std'])
    ])

def get_val_transforms():
    """Returns transformations for the validation dataset."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VAL_TRANSFORMS_CONFIG['normalize_mean'], std=VAL_TRANSFORMS_CONFIG['normalize_std'])
    ])

def get_dataloaders(train_dir, val_dir, batch_size):
    """Creates and returns the training and validation DataLoaders."""
    train_dataset = CustomImageFolder(root_dir=train_dir, transform=get_train_transforms())
    val_dataset = CustomImageFolder(root_dir=val_dir, transform=get_val_transforms())
    
    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, class_names


# =============================================================================
# Part 4: TRAINING AND VALIDATION LOOPS
# =============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one full training pass over the dataset."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
        progress_bar.set_postfix(
            loss=f"{(running_loss / total_samples):.4f}",
            acc=f"{(correct_predictions.double() / total_samples):.4f}"
        )
        
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    """Performs one full validation pass over the dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Validation", total=len(dataloader), leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            progress_bar.set_postfix(
                loss=f"{(running_loss / total_samples):.4f}",
                acc=f"{(correct_predictions.double() / total_samples):.4f}"
            )

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    return epoch_loss, epoch_acc


# =============================================================================
# Part 5: MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, class_names = get_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE
    )
    
    num_actual_classes = len(class_names)
    
    model = build_model(num_classes=num_actual_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n----- Epoch {epoch+1}/{NUM_EPOCHS} -----")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch+1} Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1} Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ New best model saved! Accuracy: {best_val_acc:.4f}")

    end_time = time.time()
    print(f"\n--- Training Finished ---")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved at: {os.path.abspath(MODEL_SAVE_PATH)}")

