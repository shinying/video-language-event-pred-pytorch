import random

from transformers import RobertaTokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch


def linear_decay_with_warmup(step, total_steps, warmup_steps):
    if step <= warmup_steps:
        return step / warmup_steps
    else:
        return (total_steps - step) / (total_steps - warmup_steps)
    
def linear_decay(step, total_steps):
    return 1 - (step + 1) / total_steps


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
cls_tok = tokenizer.cls_token
sep_tok = tokenizer.sep_token


def collate_video(feats):
    pad_feats = pad_sequence(feats, batch_first=True)
    B, S, _ = pad_feats.shape 
    key_padding_mask = torch.ones(B, S)
    for i in range(B):
        key_padding_mask[i, :feats[i].size(0)] = 0.
    
    return {'video': pad_feats, 'mask': key_padding_mask.bool()}
    
    
def collate_text(event, subs=None):
    if subs is not None:
        return tokenizer(subs, event, padding=True, return_tensors='pt', 
                         return_attention_mask=True)
        
    # If dialogues are not considered 
    return tokenizer(event, padding=True, return_tensors='pt', 
                        return_attention_mask=True)
    
    
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
