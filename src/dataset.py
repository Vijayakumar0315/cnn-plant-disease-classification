from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# This import is correct, using a single dot to refer to config.py
# in the same 'src' package.
from . import config

def get_train_transforms():
    """
    Returns the transformations for the training dataset.
    Includes data augmentation to make the model more robust to variations.
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=config.TRAIN_TRANSFORMS['horizontal_flip_p']),
        transforms.RandomRotation(config.TRAIN_TRANSFORMS['rotation_degrees']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.TRAIN_TRANSFORMS['normalize_mean'],
            std=config.TRAIN_TRANSFORMS['normalize_std']
        )
    ])

def get_val_transforms():
    """
    Returns the transformations for the validation/testing dataset.
    No augmentation is applied here to mimic real-world prediction.
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.VAL_TRANSFORMS['normalize_mean'],
            std=config.VAL_TRANSFORMS['normalize_std']
        )
    ])

def get_dataloaders(train_dir, val_dir, batch_size):
    """
    Creates and returns the training and validation DataLoaders.
    
    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        batch_size (int): The batch size for the DataLoaders.
        
    Returns:
        tuple: A tuple containing (train_loader, val_loader, class_names).
    """
    # Create datasets using PyTorch's ImageFolder, which automatically
    # finds classes based on the folder structure.
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(root=val_dir, transform=get_val_transforms())
    
    # Get the class names from the folders (e.g., ['Bacterial', 'Fungal', ...])
    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    # Create the dataloaders that will feed batches of images to the GPU.
    # num_workers speeds up data loading by using multiple CPU cores in parallel.
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, # Shuffle training data for better learning
        num_workers=2
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False, # No need to shuffle validation data
        num_workers=2
    )
    
    return train_loader, val_loader, class_names

