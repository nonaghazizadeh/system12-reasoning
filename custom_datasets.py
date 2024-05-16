import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, df):
        self.data = df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data.iloc[idx]['input_ids'].flatten(), 
                'attention_mask': self.data.iloc[idx]['attention_mask'].flatten(),
                # 'text': self.data.iloc[idx]['text'],
                }

    # def collate_fn(self, data):        
    #     data_tensor = {}
    #     ignore_keys = {}
    #     for key in data[0].keys():
    #         # if key in ignore_keys or "text" in key:
    #         if key == "text":
    #             data_tensor[key] = [item[key] for item in data]
    #         # elif key == "labels":
    #         #     data_tensor[key] = pad_sequence(
    #         #         [torch.tensor(item[key], dtype=torch.long)
    #         #         for item in data],
    #         #         batch_first=True, padding_value=-100).to(self.args.device)
    #         else:
    #             data_tensor[key] = pad_sequence(
    #                 [torch.tensor(item[key], dtype=torch.long)
    #                 for item in data],
    #                 batch_first=True, padding_value=0).to(self.args.device)
    #     return data_tensor
    
# Initialize tokenizer
