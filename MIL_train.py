import sys
import os
import numpy as np
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score

resolution = 3

train_lib = '/notebooks/19_ZZQ/MIL-nature-medicine-2019/data_preprocess/train_' + str(resolution) + '.pki'
val_lib = '/notebooks/19_ZZQ/MIL-nature-medicine-2019/data_preprocess/val_' + str(resolution) + '.pki'
output = str(resolution)
batch_size = 32
nepochs = 100
workers = 0
test_every = 10
weights = 0.5
k = 10

best_acc = 0

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        for key in lib.keys():
            lib[key] = lib[key]
        slides = []
        for i,name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = 1
        self.size = int(np.round(224*1))
        self.level = lib['level']
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        # idxs是一个列表，我们选取了每个slide里面k个最大概率的k个tiles，然后用这个函数把它们挑出来
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = (self.grid[index][0] * 2 ** resolution, self.grid[index][1] * 2 ** resolution)
            img = self.slides[slideIDX].read_region(coord,self.level[0],(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            coord = (coord[0] * 2 ** resolution, coord[1] * 2 ** resolution)
            img = self.slides[slideIDX].read_region(coord,self.level[0],(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1) # 算出的output是二维的，也就是说每个图像的结果是一个(a, b)这样的tensor
            probs[i*batch_size:i*batch_size+input.size(0)] = output.detach()[:,1].clone() # detach及后面的把(a,b)变成a
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0] # 判断错误的除以总的判断的
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum() # 假阳性：预测为阳性且判断错误除以总共有多少阴性
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum() # 假阴性：预测为阴性且判断错误除以总共有多少阳性
    return err, fpr, fnr

def group_argtopk(groups, data, k):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k] = True
    index[:-k] = groups[k:] != groups[:-k]
    out[groups[index]] = data[index]
    return out



model = models.resnet34(True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.cuda()

if weights==0.5:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    w = torch.Tensor([1-weights,weights])
    criterion = nn.CrossEntropyLoss(w).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

cudnn.benchmark = True
#normalization
normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
trans = transforms.Compose([transforms.ToTensor(), normalize])

#load data
train_dset = MILdataset(train_lib, trans)
train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size= batch_size, shuffle=False,
    num_workers= workers, pin_memory=False)
if val_lib:
    val_dset = MILdataset(val_lib, trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)

#open output file
if str(resolution) not in os.listdir('.'):
    os.mkdir(str(resolution))
fconv = open(os.path.join(output,'convergence.csv'), 'w')
fconv.write('epoch,metric,value\n')
fconv.close()

#loop throuh epochs
for epoch in range(nepochs):
    train_dset.setmode(1)
    probs = inference(epoch, train_loader, model)
    topk = group_argtopk(np.array(train_dset.slideIDX), probs, k) #挑选出每个slide中是阳性的概率最高的k个tiles
    train_dset.maketraindata(topk)
    train_dset.shuffletraindata() # 挑出来后打乱
    train_dset.setmode(2)
    loss = train(epoch, train_loader, model, criterion, optimizer)
    print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, nepochs, loss))
    fconv = open(os.path.join(output, 'convergence.csv'), 'a')
    fconv.write('{},loss,{}\n'.format(epoch+1,loss))
    fconv.close()

    #Validation
    if val_lib and (epoch+1) % test_every == 0:
        val_dset.setmode(1)
        probs = inference(epoch, val_loader, model)
        maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        auc = roc_auc_score(val_dset.targets, pred)
        err,fpr,fnr = calc_err(pred, val_dset.targets)
        print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}\tAUC: {}'.format(epoch+1, nepochs, err, fpr, fnr, auc))
        fconv = open(os.path.join(output, 'convergence.csv'), 'a')
        fconv.write('{},error,{}\n'.format(epoch+1, err))
        fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
        fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
        fconv.write('{},auc,{}\n'.format(epoch + 1, auc))
        fconv.close()
        #Save best model
        err = (fpr+fnr)/2.
        if 1-err >= best_acc:
            best_acc = 1-err
            obj = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
            }
            torch.save(obj, os.path.join(output,'checkpoint_best.pth'))
