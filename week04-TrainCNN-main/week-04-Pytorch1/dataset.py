import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image
from torchvision import transforms as T

valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg", ".PNG", ".JPG", ".JPEG"]
with open("./class_info.json", 'r') as f:
    class2id = json.load(f)

class FoodDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        split: str, 
        transforms=None
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.totensor = T.ToTensor() 
        self.class2id = class2id
        self.data = self.prepare_dataset() #  (image 경로, class index) pair list
    
    def __len__(self): # self.data의 길이를 반환
        return len(self.data)
    
    def __getitem__(self, index):
        ##################### fill here ####################
        #   TODO: __getitem__을 정의해주세요
        ####################################################
        
        # 1. self.data를 인덱스로 참조해 image path와 class index를 받아온다.
        image_path, label = self.data[index]

        # 2. image_path에서 이미지를 읽어온다.
        image = Image.open(image_path)

        # 3. 이미지를 self.transforms에 넣어 전처리를 해준다.
        # transform: 이미지 데이터 증강
        if self.transforms:
            image = self.transforms(image) # 데이터 증강 적용
            image = self.totensor(image) # 이미지를 tensor로 변환
        else:
            image = self.totensor(image)
        
        # return 값은 dataloader에 의해 batch화 된다.
        return {
            "input": image, # shape : (ch, width, height)
            "target": label # shape : (1, )
            }
    
    def prepare_dataset(self):
        split_base = os.path.join(self.root, self.split)
        data = []
        
        for label in os.listdir(split_base):
            if label not in self.class2id:
                continue
            
            for image_name in os.listdir(os.path.join(split_base, label)):
                if os.path.splitext(image_name)[1] not in valid_images:
                    continue
                data.append((os.path.join(split_base, label, image_name), self.class2id[label]))
        
        return data