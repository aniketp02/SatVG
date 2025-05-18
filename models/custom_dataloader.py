"""
Custom dataloader for the TransVG model
Implements dataset loading for DIOR-RSVG without external dependencies
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer

class DiorRsvgDataset(Dataset):
    """
    Dataset for DIOR-RSVG visual grounding
    """
    def __init__(self, data_root, split='train', max_query_len=40, bert_model='bert-base-uncased', 
                 img_size=224, use_augmentation=False):
        """
        Initialize dataset
        
        Args:
            data_root: Path to dataset
            split: Data split ('train', 'val', 'test')
            max_query_len: Maximum query length for tokenization
            bert_model: BERT model to use for tokenization
            img_size: Image size for resizing
            use_augmentation: Whether to use data augmentation
        """
        self.data_root = data_root
        self.split = split
        self.max_query_len = max_query_len
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        
        # Set paths - match the original dataset structure
        self.img_dir = os.path.join(data_root, 'JPEGImages')
        self.anno_path = os.path.join(data_root, f'dior-{split}.pth')
        
        # Load annotations
        self.data = self._load_annotations()
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        
    def _load_annotations(self):
        """Load annotations from pickle file"""
        if not os.path.exists(self.anno_path):
            raise FileNotFoundError(f"Annotation file not found: {self.anno_path}")
        
        try:
            with open(self.anno_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} samples for {self.split} split from {self.anno_path}")
            return data
        except Exception as e:
            print(f"Error loading dataset from {self.anno_path}: {e}")
            return []
    
    def __len__(self):
        """Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item by index"""
        item = self.data[idx]
        
        # Extract data from format: [image_id, bbox, sentence]
        image_name = item[0] + '.jpg'
        bbox = item[1]  # Format: [xmin, ymin, xmax, ymax] in original image coordinates
        text = item[2]
        
        # Load image
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get original image size for bbox normalization
        orig_w, orig_h = image.size
        
        # Apply transformation (resize to img_size x img_size)
        img_tensor = self.transform(image)
        
        # Convert bbox from [xmin, ymin, xmax, ymax] to normalized [0, 1] coordinates
        x1, y1, x2, y2 = bbox
        
        # Store original bbox (pixel coordinates) for evaluation
        original_bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        
        # Normalize bbox coordinates to [0, 1] range based on original image dimensions
        norm_bbox = torch.tensor([
            x1 / orig_w,    # xmin normalized
            y1 / orig_h,    # ymin normalized
            x2 / orig_w,    # xmax normalized
            y2 / orig_h     # ymax normalized
        ], dtype=torch.float32)
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_query_len,
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract tokens and mask
        tokens = tokenized['input_ids'].squeeze(0)
        mask = tokenized['attention_mask'].squeeze(0)
        
        # Create sample dictionary
        sample = {
            'img': img_tensor,
            'text_tokens': tokens,
            'text_mask': mask,
            'target': norm_bbox,           # Normalized [0,1] coordinates for loss calculation
            'original_bbox': original_bbox, # Original pixel coordinates for evaluation
            'text': text,                  # Original text for reference
            'image_name': image_name,      # Image name for reference
            'orig_img_size': torch.tensor([orig_w, orig_h])  # Original image size for scaling
        }
        
        return sample


def collate_fn(batch):
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples
        
    Returns:
        Batched samples
    """
    # Stack tensors along batch dimension
    imgs = torch.stack([item['img'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    text_mask = torch.stack([item['text_mask'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    original_bboxes = torch.stack([item['original_bbox'] for item in batch])
    orig_img_sizes = torch.stack([item['orig_img_size'] for item in batch])
    
    # Collect text and image names
    texts = [item['text'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    
    # Create batch dictionary
    batch_dict = {
        'img': imgs,
        'text_tokens': text_tokens,
        'text_mask': text_mask,
        'target': targets,
        'original_bbox': original_bboxes,
        'text': texts,
        'image_name': image_names,
        'orig_img_size': orig_img_sizes
    }
    
    return batch_dict


def build_dataloaders(config, use_pin_memory=True):
    """
    Build data loaders for training and evaluation
    
    Args:
        config: Model configuration
        use_pin_memory: Whether to use pin_memory in DataLoader
        
    Returns:
        dataloaders: Dictionary of dataloaders
    """
    dataloaders = {}
    
    # Build training dataloader
    train_dataset = DiorRsvgDataset(
        data_root=config.data_root,
        split='train',
        max_query_len=config.max_text_len,
        bert_model=config.bert_model,
        img_size=config.image_size,
        use_augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    dataloaders['train'] = train_loader
    
    # Build validation dataloader
    val_dataset = DiorRsvgDataset(
        data_root=config.data_root,
        split='val',
        max_query_len=config.max_text_len,
        bert_model=config.bert_model,
        img_size=config.image_size,
        use_augmentation=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    dataloaders['val'] = val_loader
    
    # Build test dataloader
    test_dataset = DiorRsvgDataset(
        data_root=config.data_root,
        split='test',
        max_query_len=config.max_text_len,
        bert_model=config.bert_model,
        img_size=config.image_size,
        use_augmentation=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    dataloaders['test'] = test_loader
    
    return dataloaders 