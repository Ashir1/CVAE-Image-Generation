import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from autoencoder import ConditionalAutoencoderKL
from coco_dataset import CocoDataset
import os

# Load the model checkpoint
n_iter =46
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
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# val_dataset = CocoDataset(root_dir='./coco/images/train2017', annotation_file='./coco/annotations/captions_train2017.json', transform=transform)
val_dataset = CocoDataset(root_dir='./coco/images/val2017', annotation_file='./coco/annotations/captions_val2017.json', transform=transform)

# Get a sample from the validation set
sample = val_dataset[0]
image = sample['image'].unsqueeze(0).to(device)
caption = sample['caption'].to(device)

# Generate an image with the model
with torch.no_grad():
    reconstructions= model.sample(text_emb=caption)


generated_image = reconstructions.squeeze(0).cpu().permute(1, 2, 0).numpy()
generated_image = (generated_image + 1) / 2

generated_image = (generated_image * 255).astype('uint8')
generated_image = Image.fromarray(generated_image)
generated_image.save('generated_image.png')

print(f"Generated image saved as 'generated_image.png' with caption: {sample['text']}")