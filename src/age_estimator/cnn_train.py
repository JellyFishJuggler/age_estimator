import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image

class AgeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        super().__init__()
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        imgPath = row['file_path']
        img = Image.open(imgPath).convert('RGB')

        age = torch.tensor(row['age'], dtype=torch.float32)

        if self.transform:
            image = self.transform(img)
        
        return image, age

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = AgeDataset(csv_file='src/age_estimator/data/dataset.csv', transform=train_transform)
    train_size = int(len(dataset) * 0.9)
    dev_size = len(dataset) - train_size
    train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])

    # Parallel loading ke liye num_workers=2 aur pin_memory=True
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    epochs = 15
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device).unsqueeze(1)

            optimizer.zero_grad()
            op = model(images)
            loss = criterion(op, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, ages in dev_loader:
                images, ages = images.to(device), ages.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, ages)
                running_val_loss += loss.item() * images.size(0)
                
        val_loss = running_val_loss / len(dev_dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train MAE: {epoch_loss:.4f} years | Val MAE: {val_loss:.4f} years")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_age_model.pth")
            print("--> Saved Best Model Checkpoint")

    print("Training Complete!")