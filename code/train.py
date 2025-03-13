import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.colorization_cnn import ZhangColorizationNet
from data.dataset import LabColorizationDataset

# Hyperparameters
batch_size = 32
epochs = 50
learning_rate = 0.001

# Dataset and DataLoader
dataset = LabColorizationDataset('./dataset/4/l/', './dataset/4/ab/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = ZhangColorizationNet().to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load checkpoint if available
try:
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print("Checkpoint loaded successfully.")
except FileNotFoundError:
    start_epoch = 0
    print("No checkpoint found, starting training from scratch.")

# Training loop
for epoch in range(start_epoch, epochs):
    for i, (L, ab) in enumerate(dataloader):
        L, ab = L.to(device), ab.to(device)

        optimizer.zero_grad()
        L = L.float()

        # Model forward pass
        output = model(L)

        # Resize ab channels to match the output size
        ab_resized = F.interpolate(ab, size=(output.shape[2], output.shape[3]), mode='bilinear', align_corners=False)

        # Loss calculation
        loss = criterion(output, ab_resized)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.8f}')

    # Save checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, 'checkpoint.pth')
    print("Checkpoint saved!")
