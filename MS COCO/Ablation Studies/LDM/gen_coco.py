import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from autoencoder import ConditionalAutoencoderKL
from coco_dataset import CocoDataset
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
# Load the model checkpoint
n_iter =20
checkpoint_path = './cvae/autoencoder_kl_'+str(n_iter)+'.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model configuration
ddconfig = {
      "double_z": False,
      "z_channels": 3,
      "resolution": 128,
      "in_channels": 3,
      "out_ch": 3,
      "ch": 128,
      "ch_mult": [ 1,2,4 ] , # num_down = len(ch_mult)-1
      "num_res_blocks": 2,
      "attn_resolutions": [ ],
      "dropout": 0.0,
}

# Initialize the model
model = ConditionalAutoencoderKL(ddconfig=ddconfig).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Load the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)

# Load the validation dataset
# Load the validation dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_dataset = CocoDataset(root_dir='./coco/images/val2017', annotation_file='./coco/annotations/captions_val2017.json', transform=transform)

# Generate 5 images with captions
for i in range(5):
    sample = val_dataset[i]
    image = sample['image'].unsqueeze(0).to(device)
    caption = sample['caption'].to(device)  # Ensure caption is [1, 768]

    # Generate an image with the model
    with torch.no_grad():
        reconstructions = model.sample(text_emb=caption)

    generated_image = reconstructions.squeeze(0).cpu().permute(1, 2, 0).numpy()
    generated_image = (generated_image + 1) / 2
    generated_image = (generated_image * 255).astype('uint8')
    generated_image = Image.fromarray(generated_image)

    # Add caption text on top of the image
    draw = ImageDraw.Draw(generated_image)
    font = ImageFont.load_default()
    text = sample['text']
    max_width = generated_image.width
    wrapped_text = textwrap.fill(text, width=40)  # Adjust width as needed

    # Calculate text height
    lines = wrapped_text.split('\n')
    text_height = sum([draw.textsize(line, font=font)[1] for line in lines])

    # Draw a rectangle for the text background
    draw.rectangle([(0, 0), (max_width, text_height)], fill="black")

    # Draw the text
    y_text = 0
    for line in lines:
        draw.text((0, y_text), line, fill="white", font=font)
        y_text += draw.textsize(line, font=font)[1]

    # Save the image
    generated_image.save(f'generated_image_{i}.png')
    print(f"Generated image saved as 'generated_image_{i}.png' with caption: {text}")