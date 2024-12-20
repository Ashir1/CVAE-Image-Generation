import os
import json
import random
from PIL import Image
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, fraction=1.0):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotation_file (str): Path to the COCO annotation JSON file.
            transform (callable, optional): Optional transform to be applied on an image.
            fraction (float): Fraction of the dataset to use (default: 1.0).
        """
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.image_id_to_annotations = self._create_image_id_to_annotations()
        self._sample_fraction(fraction)

    def _create_image_id_to_annotations(self):
        image_id_to_annotations = {}
        for annotation in self.annotations:
            image_id = annotation['image_id']
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(annotation)
        return image_id_to_annotations

    def _sample_fraction(self, fraction):
        num_samples = int(len(self.images) * fraction)
        sampled_images = random.sample(self.images, num_samples)
        sampled_image_ids = {img['id'] for img in sampled_images}
        self.images = sampled_images
        self.annotations = [ann for ann in self.annotations if ann['image_id'] in sampled_image_ids]
        self.image_id_to_annotations = self._create_image_id_to_annotations()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: A sample containing:
                - image (Tensor): Transformed image tensor.
                - caption (Tensor): Text embedding from BERT.
        """
        # Get image, annotation pair
        image_info = self.images[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get random caption
        captions = self.image_id_to_annotations[image_id]
        caption = random.choice(captions)['caption']

        # Tokenize the caption and compute BERT embeddings
        inputs = tokenizer(caption, return_tensors='pt', add_special_tokens=True, padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        masked_token_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
        text_embedding = masked_token_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        # Return the image and its caption embedding
        sample = {
            'image': image,
            'caption': text_embedding.squeeze(0)  
        }
        return sample