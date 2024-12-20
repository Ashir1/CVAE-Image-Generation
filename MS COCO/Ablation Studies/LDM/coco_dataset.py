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
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.image_id_to_annotations = self._create_image_id_to_annotations()

        # if fraction < 1.0:
        self._sample_fraction(0.05)

    def _sample_fraction(self, fraction):
        num_samples = int(len(self.images) * fraction)
        sampled_images = random.sample(self.images, num_samples)
        sampled_image_ids = {img['id'] for img in sampled_images}

        self.images = sampled_images
        self.annotations = [ann for ann in self.annotations if ann['image_id'] in sampled_image_ids]
        self.image_id_to_annotations = self._create_image_id_to_annotations()


    def _create_image_id_to_annotations(self):
        image_id_to_annotations = {}
        for annotation in self.annotations:
            image_id = annotation['image_id']
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(annotation)
        return image_id_to_annotations
    


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Randomly select one caption for the image
        captions = self.image_id_to_annotations[image_id]
        caption = captions[0]['caption']
        
        inputs = tokenizer(caption, return_tensors='pt', add_special_tokens=True)

        with torch.no_grad():
            outputs = bert(**inputs)

        last_hidden_state = outputs.last_hidden_state

        cls_embedding = last_hidden_state[0][0]

        token_embeddings = last_hidden_state[0]
        attention_mask = inputs['attention_mask'][0]
        masked_token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  #mask padding tokens
        text_embedding = masked_token_embeddings.sum(dim=0) / attention_mask.sum()  #average over all tokens
        
        sample = {'image': image, 'caption': text_embedding, 'text': caption}
        return sample

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(root_dir='./coco/images/train2017', annotation_file='./coco/annotations/captions_train2017.json', transform=transform)
    print(dataset[0])