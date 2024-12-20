import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import json
import random
from coco_dataset import CocoDataset  
from models.baseline_cvae import BaselineConvVAE 
from utils import *  # Replace with your sample function

def main():
    root_dir = "coco/images/train2017"  
    annotation_file = "coco/images/annotations/captions_train2017.json" 
    model_path = "model_weights/vae_weights_iter19.pth"
    output_dir = "samples"  
    num_samples_per_caption = 5  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),  
    ])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device)

    dataset = CocoDataset(root_dir=root_dir, annotation_file=annotation_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    vae = load_model(model_path, device)
    
    os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        caption_embedding = batch['caption'].to(device)  
        caption_text = dataset.annotations[i]['caption']  

        results = sample_from_decoder(
            vae=vae,
            annotations=[caption_text],
            bert=bert,
            tokenizer=tokenizer,
            num_samples=num_samples_per_caption,
            device=device
        )

        generated_images = [result[0] for result in results]

        grid_path = os.path.join(output_dir, f"caption_{i}_grid.png")
        create_caption_grid(generated_images, caption_text, grid_path)
        print(f"Captioned grid saved to {grid_path}")

        if i >= 10:  
            break

if __name__ == "__main__":
    main()