import os
import torch
import torchbearer
import h5py
import pickle

import torch.nn.functional as F
import numpy               as np

from time                                  import time
from torchbearer                           import Trial
from torchbearer                           import callbacks
from torch                                 import nn
from torchbearer.callbacks.torch_scheduler import LambdaLR
from torchbearer.callbacks                 import Callback
from torch.utils.data                      import Dataset, DataLoader
from sklearn.svm                           import LinearSVC
from sklearn.metrics                       import accuracy_score
from tqdm                                  import tqdm


class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        torch.nn.init.constant_(self.transform.weight, 0)
        torch.nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda:0')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

 
class DGCNN_partseg(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, seg_num_all=27):
        super(DGCNN_partseg, self).__init__()
        self.seg_num_all   = seg_num_all
        self.k             = k
        self.transform_net = Transform_Net()
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, t):
        x, l = t
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x.permute(0,2,1)


class H5Dataset(Dataset):
    def __init__(self, files_path, numpoints=1024):
        with open(files_path, 'r') as f:
            h5files = f.read().strip().split('\n')

        size = 0

        modelsdsets, labelsdsets = [], []

        for h5file in h5files:
            dataset = h5py.File(h5file, 'r')

            size += dataset['label'].shape[0]
            
            modelsdsets.append(dataset['data'])
            labelsdsets.append(dataset['label'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        models, labels = torch.empty(size, 3, numpoints).to(device), torch.empty(size, dtype=torch.long).to(device)

        acc = 0

        for modelsdset, labelsdset in zip(modelsdsets, labelsdsets):
            npmodels = np.asarray(modelsdset)
            nplabels = np.asarray(labelsdset)

            numelems = npmodels.shape[0]

            np.apply_along_axis(np.random.shuffle, 1, npmodels) # shuffle points

            models[acc:acc+numelems] = torch.as_tensor(npmodels[:, :numpoints, :]).permute(0, 2, 1).float()
            labels[acc:acc+numelems] = torch.as_tensor(nplabels.squeeze()).long()

            acc += numelems

        self.models = models
        self.labels = labels
        self.size   = size

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f'self.__class__.__name__ index out of range.')

        return (self.models[index], torch.zeros(16, 1).to(self.device)), self.labels[index]

    def __len__(self):
        return self.size


def get_embeddings(model, layer, X, batch_size):
    embeddings = torch.empty(batch_size, 1024, 1024)

    def copy_data(m, i, o):
        embeddings.copy_(o.data)

    hook = layer.register_forward_hook(copy_data)

    model(X)

    hook.remove()

    return (embeddings.max(dim=-1, keepdim=True)[0]).numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DGCNN_partseg().to(device)

trainloader = DataLoader(H5Dataset('train_files.txt'), batch_size=32, shuffle=True)
testloader  = DataLoader(H5Dataset('test_files.txt'), batch_size=32, shuffle=True)

with open('acc2048.txt', 'w') as f:
    f.write('Epochs, Accuracy\n')

for epochs in range(0, 40, 10):
    checkpoint = torch.load(f'states/pretrained_sn_{epochs}.state')

    model.load_state_dict(checkpoint['model'])

    layer = model._modules.get('conv6')

    model.eval()

    svc = LinearSVC(class_weight='balanced', max_iter=5e4)

    count = len(trainloader.dataset)

    embeddings, embeddings_labels = np.empty((count, 1024, 1)), np.empty((count))

    acc = 0

    for tuples, labels in tqdm(trainloader):
        preds = get_embeddings(model, layer, tuples, labels.shape[0])
        
        embeddings[acc:acc+preds.shape[0]]        = preds
        embeddings_labels[acc:acc+preds.shape[0]] = labels.cpu().numpy()

        acc += preds.shape[0]

    print('Fitting SVM...')
    svc.fit(embeddings.squeeze(2), embeddings_labels)
    print('Done.')

    count = len(testloader.dataset)

    embeddings, embeddings_labels = np.empty((count, 1024, 1)), np.empty((count))

    acc = 0

    for tuples, labels in tqdm(testloader):
        preds = get_embeddings(model, layer, tuples, labels.shape[0])

        embeddings[acc:acc+preds.shape[0]]        = preds
        embeddings_labels[acc:acc+preds.shape[0]] = labels.cpu().numpy()

        acc += preds.shape[0]

    accuracy = svc.score(embeddings.squeeze(2), embeddings_labels) * 100

    print(f'Accuracy at {epochs} epochs: {accuracy:.4f}%')

    with open('acc2048.txt', 'a') as f:
        f.write(f'{epochs}, {accuracy:.4f}\n')
