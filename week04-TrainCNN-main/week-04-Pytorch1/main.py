import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm

from dataset import FoodDataset
from model import vanillaCNN, vanillaCNN2, VGG19

def parse_args():
    # 1. 실험 조건을 argument로 받음
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['CNN1', 'CNN2', 'VGG'], required=True, help='model architecture to train')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    # data augmentation
    transforms = T.Compose([
        T.Resize((227,227), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
    ])

    # 2. dataset, dataloader를 정의
    train_dataset = FoodDataset("./data", "train", transforms=transforms) # 데이터셋 구성 및 데이터 증강 적용
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True) # 배치화
    val_dataset = FoodDataset("./data", "val", transforms=transforms) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 3. 모델 정의 
    if args.model == 'CNN1':
        model = vanillaCNN()
    elif args.model == 'CNN2':
        model = vanillaCNN2()
    elif args.model == 'VGG': 
        model = VGG19()
    else:
        raise ValueError("model not supported")
        
    ##########################   fill here   ###########################
        
    # TODO : Training Loop을 작성해주세요
    # 1. logger, optimizer, criterion(loss function)을 정의합니다.
    # train loader는 training에 val loader는 epoch 성능 측정에 사용됩니다.
    # torch.save()를 이용해 epoch마다 model이 저장되도록 해 주세요
            
    ######################################################################
    
    # 4. optimizer와 loss function 정의
    logging.basicConfig(filename=f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}/log.txt', level=logging.DEBUG)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # model train

    for epoch in range(args.epoch):
        # training mode로 전환
        # 전환되지 않으면 dropout layer가 잘못 적용될 수 있음
        model.train()
        total_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/ {args.epoch}', unit='batch')):
            # data: batch data
            inputs = data['input']
            labels = data['target']
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # 배치 마다의 loss logging
            logging.debug(f"Step {i} loss: {loss.item()}")

            total_loss += loss.item()
        
        # epoch 하다 끝남
        # len(train_loader) = 전체 데이터에 대한 총 batch의 개수
        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} average loss: {avg_train_loss}")

        # evaluation mode로 전환
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(): # autograd engine의 gradient 추적 막기
            for data in tqdm(val_loader, desc=f'Validation', unit='batch'):
                inputs = data['input']
                labels = data['target']
                output = model(inputs) 
                # 모델의 output에서 각 미니배치의 예측 클래스 결정
                # 가장 높은 확률을 가진 class index를 반환
                _, predicted = torch.max(output, 1) # 최대값 및 최대값의 인덱스 반환
                # labels: 1차원, batch size만큼의 원소를 가짐
                # total: 전체 데이터의 개수로, 현재 batch의 데이터 개수를 누적해준다. 
                total += labels.size(0) # = batch size
                correct += (predicted == labels).sum().item() # 예측값과 실제값이 같은 경우의 개수를 누적

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} accuracy = {accuracy}")
        
        # epoch마다 모델 저장
        save_dir = f"./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}/"
        filename = f"{epoch}_score:{accuracy:.3f}.pth"
        torch.save(model.state_dict(), save_dir + filename)

