import math

from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout
from transformers import RobertaModel
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe / torch.linalg.norm(pe, dim=2, keepdim=True)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
      
        
class VLEPModel(nn.Module):
    
    def __init__(self, input_size=2048, hidden_size=768, max_len=256, dropout=0.1, 
                 nhead=12, learned_pos_embed=True):
        super().__init__()
        self.vid_project = nn.Linear(input_size, hidden_size, bias=False)
        if learned_pos_embed:
            self.pos_encoding = nn.Embedding(max_len, hidden_size)
        else:
            self.pos_encoding = PositionalEncoding(hidden_size, dropout, max_len)
        
        self.vid_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, 
                                                      dropout=dropout, activation='gelu')

        self.text_encoder = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        self.cross_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead,
                                                        dropout=dropout, activation='gelu')
        
        self.pooler = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.head = nn.Linear(768, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.vid_project.weight)
        nn.init.normal_(self.pos_encoding.weight, mean=0., std=0.02)
        
        nn.init.xavier_uniform_(self.vid_encoder.self_attn.in_proj_weight, gain=2**-0.5)
        nn.init.constant_(self.vid_encoder.self_attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.cross_encoder.self_attn.in_proj_weight, gain=2**-0.5)
        nn.init.constant_(self.cross_encoder.self_attn.out_proj.bias, 0.)
        
        nn.init.constant_(self.head.bias, 0.)
        
    def forward(self, batch):
        """
        batch contains vid_feats, vid_masks, text_ids, text_masks
        """
        v = self.vid_project(batch['vid_feats']) # batch, seq, hidden
        v = (v + self.pos_encoding(torch.arange(v.size(1), device=v.device))).transpose(0, 1) # seq, batch, hidden
        v = self.vid_encoder(src=v, src_key_padding_mask=batch['vid_masks'])
        
        t1 = self.text_encoder(input_ids=batch['text1_ids'], attention_mask=batch['text1_masks']).last_hidden_state
        t2 = self.text_encoder(input_ids=batch['text2_ids'], attention_mask=batch['text2_masks']).last_hidden_state
       
	    # For dialogue-only 
        # t1 = torch.cat([t1.pooler_output.unsqueeze(1), t1.last_hidden_state[:, 1:]], dim=1)
        # t2 = torch.cat([t2.pooler_output.unsqueeze(1), t2.last_hidden_state[:, 1:]], dim=1)
        # x = torch.stack([t1, t2], dim=1)
        
        m1 = torch.cat([~(batch['text1_masks'].bool()), batch['vid_masks']], dim=1)
        m2 = torch.cat([~(batch['text2_masks'].bool()), batch['vid_masks']], dim=1)
        x1 = self.cross_encoder(src=torch.cat([t1.transpose(0, 1), v], dim=0), 
                                src_key_padding_mask=m1)[0]
        x2 = self.cross_encoder(src=torch.cat([t2.transpose(0, 1), v], dim=0), 
                                src_key_padding_mask=m2)[0]
        x = torch.stack([x1, x2], dim=1)

        return self.head(x).squeeze()
    
    def set_roberta_grad(self, requires_grad):
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = requires_grad
        for param in self.text_encoder.encoder.layer[:-1].parameters():
            param.requires_grad = requires_grad
            
