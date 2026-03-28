import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_config(name: str) -> dict:
    """Dimensionality and class specifications for each manifold."""
    cfg = {
        "fashion-mnist": {"dim": 28*28*3, "classes": 10, "names": ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']},
        "stl-10": {"dim": 96*96*3, "classes": 10, "names": ['Air', 'Bird', 'Car', 'Cat', 'Deer', 'Dog', 'Horse', 'Monk', 'Ship', 'Trck']},
        "eurosat": {"dim": 64*64*3, "classes": 10, "names": ['Crop', 'Forest', 'Herb', 'High', 'Indus', 'Past', 'Perm', 'Resi', 'River', 'Lake']}
    }
    return cfg.get(name)

def get_loaders(name: str, batch_size: int = 128):
    """
    Industrial-grade loader: checks local cache, downloads if missing.
    Ensures uniformity by upscaling Grayscale to RGB.
    """
    transform = transforms.Compose([
        # Vectorizing Grayscale to RGB to maintain uniform linear architecture
        transforms.Grayscale(3) if name == "fashion-mnist" else transforms.Lambda(lambda x: x),
        transforms.Resize((64, 64)) if name == "eurosat" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    data_path = './data'
    os.makedirs(data_path, exist_ok=True)

    if name == "stl-10":
        train_set = datasets.STL10(data_path, split='train', download=True, transform=transform)
        test_set = datasets.STL10(data_path, split='test', download=True, transform=transform)
    
    elif name == "eurosat":
        # Auto-Check & Download Logic for heavy satellite data
        try:
            # Attempt automated download via torchvision
            full_dataset = datasets.EuroSAT(data_path, download=True, transform=transform)
            print(f"[SYSTEM] EuroSAT data stream stabilized.")
        except Exception:
            # Fallback: Check if the folder was manually placed
            local_dir = os.path.join(data_path, '2750')
            if os.path.exists(local_dir):
                full_dataset = datasets.ImageFolder(local_dir, transform=transform)
                print(f"[SYSTEM] EuroSAT localized internally at {local_dir}")
            else:
                print(f"[CRITICAL] EuroSAT missing. Automated download failed.")
                raise FileNotFoundError("Check internet connection or place EuroSAT manually in ./data/2750")

        # Split 80/20 since EuroSAT lacks a default train/test split
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_set, test_set = random_split(full_dataset, [train_size, test_size])
    
    else: 
        # Default fallback: Fashion-MNIST
        train_set = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    config = get_data_config(name)
    return train_loader, test_loader, config["dim"], config["classes"]