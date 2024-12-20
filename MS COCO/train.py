import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import os

from coco_dataset import CocoDataset
from models.baseline_cvae import BaselineConvVAE
from utils import sample_and_save

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

save_dir = "model_weights"
os.makedirs(save_dir, exist_ok=True)

dataset = CocoDataset(
    root_dir='coco/images/train2017',
    annotation_file='coco/images/annotations/captions_train2017.json',
    transform=transform,
    fraction=0.3
)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

latent_dim = 128
text_embedding_dim = 768  # BERT embedding size
cvae = BaselineConvVAE(text_embedding_dim=text_embedding_dim, latent_dim=latent_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cvae = cvae.to(device)
optimizer = optim.Adam(cvae.parameters(), lr=1e-4)
num_epochs = 20

os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

with open(log_file, "w") as f: 
    f.write("Epoch, Total Loss\n") 

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        captions_batch = None  

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = batch['image'].to(device)  # Shape: [batch_size, 3, 64, 64]
            captions = batch['caption'].to(device)  # Shape: [batch_size, 768]

            if captions_batch is None:
                captions_batch = captions

            optimizer.zero_grad()

            outputs = cvae(images, captions)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        f.write(f"{epoch + 1}, {avg_loss:.4f}\n")
        f.flush()  

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} ")

        # Save a sampled image every other epoch
        if (epoch + 1) % 1 == 0:
            sample_and_save(cvae, captions_batch, epoch + 1, device)
        
        torch.save(cvae.state_dict(), os.path.join(save_dir, f"vae_weights_iter{epoch}.pth"))