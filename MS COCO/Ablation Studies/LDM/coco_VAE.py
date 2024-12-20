import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from coco_dataset import CocoDataset
from model import Encoder, Decoder
from model import ResnetBlock, Downsample, Upsample, make_attn
import wandb
from transformers import BertTokenizer, BertModel
from util import instantiate_from_config
from contperceptual import LPIPSLoss, VAELoss, VAELossorg
from autoencoder import ConditionalAutoencoderKL
import yaml

#Load the configuration from a YAML file.
def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load config from LDM VAE
yaml_path = './autoencoder/autoencoder_kl_64x64x3.yaml'
config = load_config_from_yaml(yaml_path)

# Extract model configuration
model_config = config['model']
ddconfig = model_config['params']['ddconfig']
lossconfig = model_config['params']['lossconfig']
embed_dim = model_config['params']['embed_dim']
monitor = model_config['params']['monitor']

def get_dataloader(batch_size, img_size, train_img_dir, test_img_dir, train_annotation_file, val_annotation_file):
    # Define the transformations, normalize to [-1,1]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the training and validation datasets
    train_dataset = CocoDataset(root_dir=train_img_dir, annotation_file=train_annotation_file, transform=transform)
    val_dataset = CocoDataset(root_dir=test_img_dir, annotation_file=val_annotation_file, transform=transform)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Training loop
def train(model, train_loader, val_loader, device, epochs, learning_rate, use_wandb=False):
    loss_fn = VAELossorg().to(device) 
    optimizer = optim.Adam(list(model.parameters()) 
                           ,lr=learning_rate, betas=(0.5, 0.9),weight_decay=1e-5)
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            images = data['image'].to(device)
            text_emb = data['caption'].to(device)
            reconstructions, mean, logvar = model(images, text_emb=text_emb)
            optimizer.zero_grad()
            loss, log = loss_fn(images, reconstructions, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if use_wandb:
                wandb.log({
                    # "batch_idx": batch_idx,
                    "total_loss": log["total_loss"].item(),
                    # "logvar": log["logvar"].item(),
                    "kl_loss": log["kl_loss"].item(),
                    # "nll_loss": log["nll_loss"].item(),
                    "rec_loss": log["rec_loss"].item()
                })

        avg_loss = total_loss / len(train_loader)
        if use_wandb:
            wandb.log({"train Epoch Loss": avg_loss})
        validate(model, val_loader, device, use_wandb = use_wandb)

        if epoch >= 20:
            model_save_path = f"./cvae/autoencoder_kl_{epoch}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

def validate(model, val_loader, device, use_wandb=False):
    model.eval()
    total_loss = 0
    loss_fn = VAELossorg().to(device) 
    loss_fn.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            images = data['image'].to(device)
            text_emb = data['caption'].to(device)
            
            reconstructions, mean, logvar = model(images, text_emb=text_emb)

            loss, log = loss_fn(images, reconstructions, mean, logvar)
            total_loss += loss.item()
            
            if use_wandb:
                wandb.log({
                    # "batch_idx": batch_idx,
                    "val_total_loss": log["total_loss"].item(),
                    # "logvar": log["logvar"].item(),
                    "val_kl_loss": log["kl_loss"].item(),
                    # "nll_loss": log["nll_loss"].item(),
                    "val_rec_loss": log["rec_loss"].item()
                })

    avg_loss = total_loss / len(val_loader)
    if use_wandb:
        wandb.log({"val Epoch Loss": avg_loss})
    print(f"Validation Loss: {avg_loss}")

# Main function
def main():
    img_size = 128
    batch_size = 16  # Adjust the batch size as needed
    epochs = 100
    learning_rate = 1e-4
    use_wandb = True


    if not os.path.exists('cvae'):
        os.makedirs('cvae')

    if use_wandb:
        wandb.init(project="CVAE_2")

    # Set the device to 'cuda' if GPUs are available
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = ConditionalAutoencoderKL(ddconfig=ddconfig).to(device)
    
    # Prepare data loaders
    train_loader, val_loader = get_dataloader(batch_size, img_size, train_img_dir='./coco/images/train2017', test_img_dir='./coco/images/val2017', train_annotation_file='./coco/annotations/captions_train2017.json', val_annotation_file='./coco/annotations/captions_val2017.json')

    # Start training
    train(model, train_loader, val_loader, device, epochs, learning_rate, use_wandb=use_wandb)

if __name__ == '__main__':
    main()