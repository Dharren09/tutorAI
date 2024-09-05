import lightning as L
from lightning.data import LightningDataModule
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
import pandas as pd 
import torch
from dataset import load_dataset, load_from_disk
from Typing import dict, List, Union, Callable


class TextDataModule(LightningDataModule):
    def __init__(
        self, batch_size: int = 16,
        text_datasets: List[str] = None,
        max_length: int = None
    ) -> LightningDataModule:
    
        super(TextDataModule, self).__init__()
        self.datasets_dict = load_textData(text_datasets)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.sentence_transformer = SentenceTransformer('all_MiniLM-L6-v2')

    def load_textData(self, text_datasets):
        # load datasets
        datasets_dict = defaultdict()
        for item in self.text_datasets:
            dataset = load_dataset(item, trust_remote_code=True, streaming=True)
            # dataset.save_to_disk('tutorAI/data/text_data/{item}')
            datasets_dict[item] = dataset
        return datasets_dict

    def setup(self, stage=None):
        # preprocess text data
        self.train_data = []
        self.val_data = []
        
        for dataset in self.datasets_dict.values():
            dataset['text'] = dataset['text'].apply(
                lambda x: self.tokenizer.encode(x, max_length=self.max_length, padding='max_length', truncation=True, convert_to_tensor=True))
            dataset['embedding'] = dataset['embedding'].apply(
                lambda x: self.sentence_transformer.encode(x))
            dataset['label'] = torch.tensor(dataset['label'])
            
        # splitting the data into train and val sets
        self.train_dataset, self.val_dataset = dataset.split(test_size=0.2, random_state=42)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)