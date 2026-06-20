# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append("..")
from cmath import nan
import numpy as np
import pickle

import sys,os,random
from importlib import import_module
import torch
import torch.nn as nn
import torch.utils.data
import argparse

from models.HH_conv import SCNN_Model_HH
from models.HH_fc import SNN_Model_HH
from models.LIF_hh_conv import SCNN_Model_LIF_hh
from models.LIF_hh_fc import SNN_Model_LIF_hh
from models.LIF_conv import SCNN_Model_LIF
from models.LIF_fc import SNN_Model_LIF
from models.fourLIF_conv import SCNN_Model_4LIF
from models.fourLIF_fc import SNN_Model_4LIF
from models.ann import ANN_Model
from models.cnn import CNN_Model
from models.models_4LIF_fc.LIF_1_3_fc import SNN_Model_LIF_1_3
from models.models_4LIF_fc.LIF_2_2_fc import SNN_Model_LIF_2_2
from models.models_4LIF_fc.LIF_1_2_1_fc import SNN_Model_LIF_1_2_1
from models.models_4LIF_fc.LIF_1_1_1_1_fc import SNN_Model_LIF_1_1_1_1
from models.models_4LIF_fc.LIF_ring_fc import SNN_Model_LIF_ring
from models.models_4LIF_conv.LIF_1_3_conv import SCNN_Model_LIF_1_3
from models.models_4LIF_conv.LIF_2_2_conv import SCNN_Model_LIF_2_2
from models.models_4LIF_conv.LIF_1_2_1_conv import SCNN_Model_LIF_1_2_1
from models.models_4LIF_conv.LIF_1_1_1_1_conv import SCNN_Model_LIF_1_1_1_1
from models.models_4LIF_conv.LIF_ring_conv import SCNN_Model_LIF_ring
from models.models_5LIF_fc.LIF_1_3_1_fc import SNN_Model_LIF_1_3_1
from models.models_5LIF_fc.LIF_1_1_3_fc import SNN_Model_LIF_1_1_3
from models.models_5LIF_fc.LIF_1_2_2_fc import SNN_Model_LIF_1_2_2
from models.models_5LIF_fc.LIF_1_1_2_1_fc import SNN_Model_LIF_1_1_2_1
from models.models_5LIF_fc.LIF_1_1_1_1_1_fc import SNN_Model_LIF_1_1_1_1_1
from models.models_5LIF_fc.LIF_1_4_fc import SNN_Model_LIF_1_4
from models.models_5LIF_conv.LIF_1_3_1_conv import SCNN_Model_LIF_1_3_1
from models.models_5LIF_conv.LIF_1_1_3_conv import SCNN_Model_LIF_1_1_3
from models.models_5LIF_conv.LIF_1_2_2_conv import SCNN_Model_LIF_1_2_2
from models.models_5LIF_conv.LIF_1_1_2_1_conv import SCNN_Model_LIF_1_1_2_1
from models.models_5LIF_conv.LIF_1_1_1_1_1_conv import SCNN_Model_LIF_1_1_1_1_1
from models.models_5LIF_conv.LIF_1_4_conv import SCNN_Model_LIF_1_4

device = torch.device("cuda:0")
data_path = '/data'

parser = argparse.ArgumentParser(description='multi-task')
parser.add_argument('--seed', default = 1, type=int, help='random seed')
parser.add_argument('--batch_size', default = 128, type=int, help='batch size')
parser.add_argument('--model_name',default="LIF_fc", type=str, help='LIF / HH / LIF_HH / 4LIF / LIF_1_3 / LIF_2_2 / LIF_1_2_1 / LIF_1_1_1_1 / LIF_ring / LIF_1_3_1 / LIF_1_1_3 / LIF_1_2_2 / LIF_1_1_2_1 / LIF_1_1_1_1_1 / LIF_1_4 / ANN / CNN + fc / conv')
parser.add_argument('--topk', default=5, type=int, help='top-k slots to keep for 5-LIF_fc models (default 5 = all slots; 4 = drop the slot with the lowest total spike count per sample)')
args = parser.parse_args()

def train(data_loader, model,optimizer,scheduler,criterion):     
    
    #print('==>>> total trainning batch number: {}'.format(len(data_loader)))
    model.train()
    correct0_train = 0
    correct1_train = 0
    running_loss = 0
    count = 0
          
    for (it, batch) in enumerate(data_loader):
        X = batch[0]
        ts = batch[1]
        if torch.cuda.is_available():
            X = X.to(device)
            ts = ts.to(device)  
          
        outputs = model(X) 
        task_loss = 0
        for i in range(2):
            task_output_i = outputs[:,i,:]
            task_loss += criterion(task_output_i,ts[:,i])

        task_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += task_loss.item()
        _,predict0 = torch.max(outputs[:,0,:],dim=1)
        _,predict1 = torch.max(outputs[:,1,:],dim=1)
        correct0_train += predict0.eq(ts[:,0].view_as(predict0)).sum().item()
        correct1_train += predict1.eq(ts[:,1].view_as(predict1)).sum().item()    
        count += X.size(0)
        if (it + 1) % 100 == 0:
            #print('progress:',it,'/',len(data_loader))
            #print('Loss:',running_loss,'acc1:',correct0_train/count*100,' acc2:',correct1_train/count*100)
            correct0_train = 0
            correct1_train = 0
            running_loss = 0
            count = 0
    scheduler.step() 
        
def val(data_loader, model,criterion, epoch):

    #print('==>>> total validation batch number: {}'.format(len(data_loader)))
    model.eval()
    correct0_train = 0
    correct1_train = 0
    running_loss = 0
    count = 0
    model.eval()
    with torch.no_grad(): 
        for (it, batch) in enumerate(data_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.to(device)
                ts = ts.to(device)
            outputs = model(X) 
            task_loss = 0
            for i in range(2):
                task_output_i = outputs[:,i,:]
                task_loss += criterion(task_output_i,ts[:,i])
            running_loss += task_loss.item()
            
            _,predict0 = torch.max(outputs[:,0,:],dim=1)
            _,predict1 = torch.max(outputs[:,1,:],dim=1)
            correct0_train += predict0.eq(ts[:,0].view_as(predict0)).sum().item()
            correct1_train += predict1.eq(ts[:,1].view_as(predict1)).sum().item()  
            count += X.size(0)
    print('=============Validation Start================')
    print('Epoch:',epoch,' Loss:',running_loss,' acc1:',correct0_train/count*100,' acc2:',correct1_train/count*100)
    #print('=============Validation End================')
    return correct0_train/count*100, correct1_train/count*100

def main(args):

    with open('data/multi_fashion_and_mnist.pickle','rb') as f:
        trainX, trainLabel,testX, testLabel = pickle.load(f)   
 
    trainX = torch.tensor(trainX.reshape(120000,1,36,36), dtype=torch.float32)
    trainLabel = torch.tensor(trainLabel, dtype=torch.long)
    testX = torch.tensor(testX.reshape(20000,1,36,36), dtype=torch.float32)
    testLabel = torch.tensor(testLabel, dtype=torch.long)
    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set  = torch.utils.data.TensorDataset(testX, testLabel)

    batch_size = args.batch_size         
    train_loader = torch.utils.data.DataLoader(
                        dataset=train_set,
                        batch_size=batch_size,
                        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)   

    seed_value = args.seed
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value) 
    torch.manual_seed(seed_value)                
    torch.cuda.manual_seed(seed_value)            
    torch.cuda.manual_seed_all(seed_value)         
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model_name
        
    print('model established:',model_name)
    n_tasks = 2
    if model_name == "LIF_fc":
        model = SNN_Model_LIF(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
    elif model_name == "4LIF_fc":
        model = SNN_Model_4LIF(n_tasks)   
        model.to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
    elif model_name == "LIF_1_3_fc":
        model = SNN_Model_LIF_1_3(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_2_2_fc":
        model = SNN_Model_LIF_2_2(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_2_1_fc":
        model = SNN_Model_LIF_1_2_1(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_1_1_1_fc":
        model = SNN_Model_LIF_1_1_1_1(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_ring_fc":
        model = SNN_Model_LIF_ring(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_3_1_fc":
        model = SNN_Model_LIF_1_3_1(n_tasks, topk=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_1_3_fc":
        model = SNN_Model_LIF_1_1_3(n_tasks, topk=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_2_2_fc":
        model = SNN_Model_LIF_1_2_2(n_tasks, topk=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_1_2_1_fc":
        model = SNN_Model_LIF_1_1_2_1(n_tasks, topk=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_1_1_1_1_fc":
        model = SNN_Model_LIF_1_1_1_1_1(n_tasks, topk=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "LIF_1_4_fc":
        model = SNN_Model_LIF_1_4(n_tasks, topk=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif model_name == "HH_fc":
        model = SNN_Model_HH(n_tasks)   
        model.to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
    elif model_name == "LIF_hh_fc":
        model = SNN_Model_LIF_hh(n_tasks)   
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
    elif model_name == "LIF_conv":
        model = SCNN_Model_LIF(n_tasks)   
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "4LIF_conv":
        model = SCNN_Model_4LIF(n_tasks)   
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "HH_conv":
        model = SCNN_Model_HH(n_tasks)   
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "LIF_hh_conv":
        model = SCNN_Model_LIF_hh(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "LIF_1_3_conv":
        model = SCNN_Model_LIF_1_3(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "LIF_2_2_conv":
        model = SCNN_Model_LIF_2_2(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "LIF_1_2_1_conv":
        model = SCNN_Model_LIF_1_2_1(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "LIF_1_1_1_1_conv":
        model = SCNN_Model_LIF_1_1_1_1(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "LIF_ring_conv":
        model = SCNN_Model_LIF_ring(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay = 3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "LIF_1_3_1_conv":
        model = SCNN_Model_LIF_1_3_1(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    elif model_name == "LIF_1_1_3_conv":
        model = SCNN_Model_LIF_1_1_3(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    elif model_name == "LIF_1_2_2_conv":
        model = SCNN_Model_LIF_1_2_2(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    elif model_name == "LIF_1_1_2_1_conv":
        model = SCNN_Model_LIF_1_1_2_1(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    elif model_name == "LIF_1_1_1_1_1_conv":
        model = SCNN_Model_LIF_1_1_1_1_1(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    elif model_name == "LIF_1_4_conv":
        model = SCNN_Model_LIF_1_4(n_tasks)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    elif model_name == "CNN":
        model = CNN_Model(n_tasks)   
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    elif model_name == "ANN":
        model = ANN_Model(n_tasks)   
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1) 

    Epoch = 40
    bestacc1 = 0
    bestacc2 = 0
    
    criterion = nn.CrossEntropyLoss()

    for e in range(Epoch):
        train(train_loader, model, optimizer, scheduler, criterion)
        acc1,acc2 = val(test_loader, model, criterion, e)
        bestacc1 = acc1
        bestacc2 = acc2
    print("best_acc1:", bestacc1, "best_acc2:", bestacc2)
    return bestacc1, bestacc2


main(args)