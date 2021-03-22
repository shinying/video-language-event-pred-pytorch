import glob
import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch


class VLEPDataset(Dataset):
    
    def __init__(self, vid_dir, annotation, subtitles, input_imgs=False, fps=3):
        self.vid_dir = vid_dir
        self.anno = [json.loads(line) for line in open(annotation).read().splitlines()]
        self.subs = json.load(open(subtitles))
        self.input_imgs = input_imgs
        self.fps = fps
        
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.anno)
    
    def __getitem__(self, idx):
        # Read events
        example = self.anno[idx]
        s, e = example['ts']
        
        # Read frames or frame features
        if self.input_imgs:
            files = glob.glob(os.path.join(self.vid_dir, example['vid_name'], '*.jpg'))
            files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]))
            frames = [self.transforms(Image.open(file)) for file in files]
            frames = frames[max(0, np.int(np.floor(s*self.fps))-1):np.int(np.ceil(e*self.fps))]
            vid_feat = torch.stack(frames)
        else:
            vid_feat = np.load(os.path.join(self.vid_dir, example['vid_name']+'.npy'))
            vid_feat = torch.tensor(vid_feat)
            vid_feat = vid_feat[max(0, np.int(np.floor(s))-1):np.int(np.ceil(e))]
        if len(vid_feat) == 0:
            vid_feat = vid_feat[s] 
        assert len(vid_feat) > 0
        
        # Read subtitles
        subs = [sub['text'].strip() for sub in self.subs[example['vid_name']] \
            if s <= sub['ts'][1] and e >= sub['ts'][0]]
        if len(subs) > 0:
            subs = ' '.join(subs)
        else:
            subs = ''
        
        return {'video': vid_feat, 
                'subs': subs,
                'event1': example['events'][0],
                'event2': example['events'][1],
                'label': example['answer']}
