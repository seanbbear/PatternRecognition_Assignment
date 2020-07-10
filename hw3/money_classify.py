import torch
import torch.nn as nn
from torchvision import datasets ,transforms
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image
import PIL
from torch.utils.data.sampler import SubsetRandomSampler



class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn1 = nn.Conv2d(3, 16, kernel_size=5, stride=1) 
        self.drop1 = nn.Dropout(0.2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(16,8, kernel_size=10, stride=1) 
        self.drop2 = nn.Dropout(0.2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(8 * 26 * 26, 3)     
    def forward(self, x):
        out = self.cnn1(x) 
        out = self.drop1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.drop2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out) 
        return out

def train(model,n_epochs,train_loader,test_loader,optimizer,criterion):
    train_acc_his,test_acc_his=[],[]
    train_losses_his,test_losses_his=[],[]
    train_on_gpu = torch.cuda.is_available()
    for epoch in tqdm(range(1, n_epochs+1)):
        # keep track of training and testing loss
        train_loss,test_loss = 0.0,0.0
        train_losses,test_losses=[],[]
        correct,total=0,0
        print('running epoch: {}'.format(epoch))
       
        #Training
        model.train()
        for data, target in train_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            #Calculate the batch loss
            loss = criterion(output, target)
            #Calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
            train_acc=correct/total
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item()*data.size(0))
            optimizer.zero_grad()
        #Testing 
        model.eval()
        for data, target in test_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            #Calculate the batch loss
            loss =criterion(output, target)
            #Calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
            test_acc=correct/total
            test_losses.append(loss.item()*data.size(0))
        # calculate average losses
        train_loss=np.average(train_losses)
        test_loss=np.average(test_losses)
        train_acc_his.append(train_acc)
        test_acc_his.append(test_acc)
        train_losses_his.append(train_loss)
        test_losses_his.append(test_loss)
        print('\tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(train_loss, test_loss))
        print('\tTraining Accuracy: {:.6f} \tTesting Accuracy: {:.6f}'.format(train_acc, test_acc))
    return train_acc_his,test_acc_his,train_losses_his,test_losses_his,model

def plotImage(train_losses_his,test_losses_his,train_acc_his,test_acc_his):
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    plt.plot(train_losses_his, 'bo', label = 'train loss')
    plt.plot(test_losses_his, 'r', label = 'test loss')
    plt.title("Loss")
    plt.legend(loc='upper left')
    plt.subplot(222)
    plt.plot(train_acc_his, 'bo', label = 'train accuracy')
    plt.plot(test_acc_his, 'r', label = 'test accuracy')
    plt.title("Accuracy")
    plt.legend(loc='upper left')
    plt.show()

def data_preprocessing(train_path,test_path,batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomApply(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(90,180))
            ]
        ,p=0.7),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(train_path, transform=train_transforms)
    test_data = datasets.ImageFolder(test_path, transform=test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)
    return train_loader,test_loader
    



if __name__ == "__main__":
    PATH_train = "./money/train/"
    PATH_test = "./money/test/"
    train_path = Path(PATH_train)
    test_path = Path(PATH_test)

    batch_size = 16
    LR = 5e-4
    n_epochs = 50
    
    train_loader,test_loader = data_preprocessing(train_path,test_path,batch_size)
    
    model=CNN_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    train_acc_his,test_acc_his,train_losses_his,test_losses_his,model=train(model,n_epochs,train_loader,test_loader,optimizer,criterion)

    plotImage(train_losses_his,test_losses_his,train_acc_his,test_acc_his)
    