import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# load files 
f_s = h5py.File("/home/jua/doublehiggs_jetimage/4-Dataset/dataset_hdf5/{}_dataset.hdf5".format(sys.argv[1]),"r")
f_b = h5py.File("/home/jua/doublehiggs_jetimage/4-Dataset/dataset_hdf5/{}_dataset.hdf5".format(sys.argv[2]),"r")

train_images = np.vstack((f_s["train_images"][:], f_b["train_images"][:]))
train_labels = np.hstack((f_s["train_labels"][:], f_b["train_labels"][:]))
test_images = np.vstack((f_s["test_images"][:], f_b["test_images"][:]))
test_labels = np.hstack((f_s["test_labels"][:], f_b["test_labels"][:]))
val_images = np.vstack((f_s["val_images"][:], f_b["val_images"][:]))
val_labels = np.hstack((f_s["val_labels"][:], f_b["val_labels"][:]))
classes = {0:'di-higgs',1:'ttbar'}
train_labels.astype(int)
test_labels.astype(int)
val_labels.astype(int)

train_images = torch.from_numpy(train_images)
train_labels = torch.from_numpy(train_labels)
train_sets = ([(train_images[i],train_labels[i])for i in range(len(train_labels))])

test_images = torch.from_numpy(test_images)
test_labels = torch.from_numpy(test_labels)
test_sets = ([(test_images[i],test_labels[i])for i in range(len(test_labels))])

val_images = torch.from_numpy(val_images)
val_labels = torch.from_numpy(val_labels)
val_sets = ([(val_images[i],val_labels[i])for i in range(len(val_labels))])


trainloader = torch.utils.data.DataLoader(train_sets, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_sets, batch_size=4, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_sets, batch_size=4, shuffle=True, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor= torch.max(output,1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1,4,idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
        classes[preds[idx]],
               probs[idx]*100.0,
               classes[int(labels[idx])]),
                    color=('green' if preds[idx]==labels[idx].long() else "red"))
    return fig

from datetime import datetime
now = datetime.now()
logdir = "runs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
writer = SummaryWriter(logdir)

#net = Net()
#net.cuda()

net = models.resnet18(pretrained=False)
net.conv1 = nn.Conv2d(5,64, kernel_size=(7,7),stride=(2,2),padding=(3,3), bias=False)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net = net.cuda()

since = time.time()
learning_rate = 0.001
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
model_wts = net.state_dict()
best_acc = 0.0
losses = {'train':[],'valid':[]}
accuracy = {'train':[],'valid':[]}
num_epochs = 20
for epoch in range(num_epochs): 
    print("Epoch {}/{}".format(epoch, num_epochs-1))
    print('-'*10)
    for phase in ['train', 'valid']:
      if phase =='train':
          net.train()
          loader = trainloader
      else:
          net.eval()
          loader = valloader
      running_loss = 0.0
      running_corrects= 0
      ep_running_loss = 0.0
      ep_running_corrects = 0
      for i, data in enumerate(loader, 0):
          inputs, labels = data

          inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
          #inputs, labels = Variable(inputs), Variable(labels)
  
          optimizer.zero_grad()

          outputs = net(inputs)
          _, preds = torch.max(outputs.data, 1)
          loss = criterion(outputs, labels.long())
          if phase=='train':
              loss.backward()
              optimizer.step()

          running_loss += loss.data
          ep_running_loss += loss.data
          running_corrects += torch.sum(preds==labels.long())
          ep_running_corrects += torch.sum(preds==labels.long())
          if i % 100 == 99:    # print every 100 mini-batches
              '''
              print('[%d, %5d]  loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0
              print('[%d, %5d]  Accuracy: %.3f' %
                    (epoch + 1, i + 1, float(running_corrects)/(100.0*4.0)))
              running_corrects = 0.0
              '''
              writer.add_scalar('{} loss'.format(phase),
                                     running_loss/100,
                                     epoch*len(loader)+i)
              writer.add_scalar('{} Accuracy'.format(phase),
                               float(running_corrects)/(100.0*4.0),
                               epoch*len(loader)+i)
              writer.add_figure('predictions vs actuals, {}'.format(phase),
                               plot_classes_preds(net, inputs, labels),
                               global_step=epoch*len(loader)+i)
              running_loss = 0.0
              running_corrects = 0.0
      epoch_loss = ep_running_loss/(len(loader)*4)
      epoch_acc = float(ep_running_corrects)/(len(loader)*4)
      losses[phase].append(epoch_loss)
      accuracy[phase].append(epoch_acc)
      print('{} Loss: {:.4f}, Acc : {:.4f}'.format( phase, epoch_loss, epoch_acc))    
      if phase=='valid' and epoch_acc>best_acc:
          best_acc = epoch_acc
          best_model_wts = net.state_dict()  
time_elapsed = time.time() - since
print("Training complete in {:.0f}s".format(time_elapsed//60, time_elapsed%60))
print("Best Acc: {:4f}".format(best_acc))
net.load_state_dict(best_model_wts)
print('Finished Training')

def plottingLossAcc(losses,accuracy,datatype):
  # plotting losses, and accuracy
    plt.plot(range(1,len(losses)+1),losses, 'r',label='loss')
    plt.title('{} Losses'.format(datatype.capitalize()))
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend()
    plt.savefig("plots/{}_loss.png".format(datatype))
    plt.show()

    plt.plot(range(1,len(accuracy)+1), accuracy, 'r', label='accuracy')
    plt.title('{} Accuracy'.format(datatype.capitalize()))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("plots/{}_accuracy.png".format(datatype))
    plt.show()

plottingLossAcc(losses['train'],accuracy['train'],'train')
plottingLossAcc(losses['valid'],accuracy['valid'],'valid')

correct = 0
total = 0
y_score = np.array([])
y = np.array([])
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    y_score = np.append(y_score,outputs.data.cpu().detach().numpy())
    y = np.append(y,labels.cpu().numpy())
    total += labels.size(0)
    correct += (predicted == labels.long().cuda()).sum()
    #correct += (predicted == labels.long()).sum()
    
ny = []
for i in y:
  if i == 0: ny.append([1,0])
  else : ny.append([0,1])

fpr,tpr, _  = roc_curve(np.array(ny).ravel(),y_score.ravel())
roc_auc = auc(fpr,tpr)
fig, ax = plt.subplots()
lw = 2
ax.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set(xlim=[0.0, 1.0], ylim=[0.0,1.0],title='ROC curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
fig.savefig("plots/roc.png")
plt.show()
print('Accuracy of the network on the {} test images: {} %%'.format(len(testloader),100 * correct / total))
