from argparse import Namespace
import json
import sys

from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from dataset import VLEPDataset
from model import VLEPModel
from utils import linear_decay, linear_decay_with_warmup, collate_text, collate_video
from utils import set_seed

class Worker:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, trainset, valset, model):
        loader = self._create_loader(trainset, training=True)
        model.set_roberta_grad(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.base_lr)

        base_steps = len(trainset) // self.cfg.train_batch_size * self.cfg.base_epochs
        fine_steps = len(trainset) // self.cfg.train_batch_size * self.cfg.fine_epochs
        warmup_steps = base_steps * self.cfg.warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lambda step: linear_decay_with_warmup(step, base_steps, int(base_steps*self.cfg.warmup)))

        for epoch in range(self.cfg.base_epochs+self.cfg.fine_epochs):
            model.train()
            model.zero_grad()
            if epoch == self.cfg.base_epochs:
                print('Start training Roberta')
                model.set_roberta_grad(True)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fine_lr)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step: linear_decay(step, fine_steps))
            
            train_loss = 0.
            train_acc = 0
            for i, batch in enumerate(tqdm(loader)):
                self.to_cuda(batch)
                labels = batch['labels']
                
                logits = model(batch)
                loss = self.criterion(logits, labels) / self.cfg.grad_accum
                loss.backward()
                if (i+1) % self.cfg.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                
                train_loss += loss.item()
                train_acc += (torch.argmax(logits, dim=1)==labels).float().sum().item()
                    
            train_loss /= len(loader)
            train_acc /= len(trainset)

            val_loss, val_acc = self.validate(valset, model)
            
            print(f'[Epoch {epoch+1}] train_loss: {train_loss:.4f} train_acc: {train_acc:.2%}',
                  f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.2%}')
                
    def validate(self, dataset, model):
        loader = self._create_loader(dataset, training=False)
        model.eval()
        val_loss = 0.
        val_acc = 0
        with torch.no_grad():
            for batch in tqdm(loader):
                self.to_cuda(batch)
                labels = batch['labels']
                logits = model(batch)
                loss = self.criterion(logits, labels)
                
                val_loss += loss.item() * len(labels)
                val_acc += (torch.argmax(logits, dim=1)==labels).sum().item()
                
            val_loss /= len(dataset)
            val_acc /= len(dataset)
        
        return val_loss, val_acc
    
    def analyze(self, dataset, model):
        loader = self._create_loader(dataset, training=False)
        model.eval()
        val_loss = 0.
        val_acc = 0
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader):
                self.to_cuda(batch)
                labels = batch['labels']
                logits = model(batch)
                loss = self.criterion(logits, labels)
                probs = torch.argmax(logits, dim=1)
                
                val_loss += loss.item() * len(labels)
                val_acc += (probs==labels).sum().item()
                preds.extend(probs.cpu().tolist())
                
            val_loss /= len(dataset)
            val_acc /= len(dataset)
        
        return val_loss, val_acc, preds
    
        
    def test(self, dataset, model):
        pass
    
    def to_cuda(self, batch):
        for k, v in batch.items():
            batch[k] = v.cuda()
        
    def _collate(self, inputs):
        vids = [d['video'] for d in inputs]
        subs = [d['subs'] for d in inputs] if self.cfg.train_dialogue else None
        event1 = [d['event1'] for d in inputs]
        event2 = [d['event2'] for d in inputs]
        labels = torch.tensor([d['label'] for d in inputs], dtype=torch.long)
        
        vids = collate_video(vids)
        vid_feats, vid_masks = vids['video'], vids['mask']
        texts = collate_text(event1, subs)
        text1_ids, text1_masks = texts['input_ids'], texts['attention_mask']
        texts = collate_text(event2, subs)
        text2_ids, text2_masks = texts['input_ids'], texts['attention_mask']
        
        return {'vid_feats': vid_feats, 'vid_masks': vid_masks,
                'text1_ids': text1_ids, 'text1_masks': text1_masks,
                'text2_ids': text2_ids, 'text2_masks': text2_masks,
                'labels': labels}

        
    def _create_loader(self, dataset, training=True):
        batch_size = self.cfg.train_batch_size if training \
            else self.cfg.test_batch_size
        return DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=training,
                          num_workers=self.cfg.num_workers, collate_fn=self._collate,)
        
        
        
if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    cfg = Namespace(**cfg)
    set_seed(seed=0)
    
    trainset = VLEPDataset(cfg.vid_feature, cfg.train_anno, cfg.subtitles)
    valset = VLEPDataset(cfg.vid_feature, cfg.val_anno, cfg.subtitles)
    model = VLEPModel(nhead=cfg.nhead, dropout=cfg.dropout).cuda()
    worker = Worker(cfg)
    worker.train(trainset, valset, model)
    
    torch.save(model.state_dict(), sys.argv[2])
