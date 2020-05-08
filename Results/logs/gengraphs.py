import os
import numpy as np
import matplotlib.pyplot as plt

C19files = ['C19FconvNet.txt', 'C19DownUpNet.txt', 'C19UNet.txt', 'C19UNetPlus.txt', 'C19UNetPlusPlus.txt']
LCTfiles = ['LCTFconvNet.txt', 'LCTDownUpNet.txt', 'LCTUNet.txt', 'LCTUNetPlus.txt', 'LCTUNetPlusPlus.txt']
models = ['FConvNet', 'DownUpNet', 'UNet', 'UNetPlus', 'UNetPlusPlus']

C19data = []
for file in C19files:
    f = open(file,'r')
    line = f.readline()
    fdata = []
    while line:
        if "Epoch" not in line:
            ldata = []
            line = line[:-2].split("-")[2:]
            for i in range(len(line)):
                ldata.append(float(line[i].split(":")[-1])) 
            #print(line)
            #print(ldata)
            fdata.append(ldata)
        line = f.readline()
    
    C19data.append(fdata)
C19data = np.array(C19data) # model x epoch x metric
print(C19data.shape)

LCTdata = []
for file in LCTfiles:
    f = open(file,'r')
    line = f.readline()
    fdata = []
    while line:
        if "Epoch" not in line:
            ldata = []
            line = line[:-2].split("-")[2:]
            for i in range(len(line)):
                ldata.append(float(line[i].split(":")[-1])) 
            #print(line)
            #print(ldata)
            fdata.append(ldata)
        line = f.readline()
    
    LCTdata.append(fdata)

LCTdata = np.array(LCTdata) # model x epoch x metric
print(LCTdata.shape)
###########################################
epoch = np.array([i+1 for i in range(100)])

### COVID data
T = 30
# val loss
plt.plot(epoch[:T],C19data[0,:T,3], color='b', linewidth=2.5)
plt.plot(epoch[:T],C19data[1,:T,3], color='g', linewidth=2.5)
plt.plot(epoch[:T],C19data[2,:T,3], color='m', linewidth=2.5)
plt.plot(epoch[:T],C19data[3,:T,3], color='c', linewidth=2.5)
plt.plot(epoch[:T],C19data[4,:T,3], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('COVID-19 Learning Curve')
plt.savefig('./C19_plots/C19_val_loss.png')
plt.clf()

# train loss
plt.plot(epoch[:T],C19data[0,:T,0], color='b', linewidth=2.5)
plt.plot(epoch[:T],C19data[1,:T,0], color='g', linewidth=2.5)
plt.plot(epoch[:T],C19data[2,:T,0], color='m', linewidth=2.5)
plt.plot(epoch[:T],C19data[3,:T,0], color='c', linewidth=2.5)
plt.plot(epoch[:T],C19data[4,:T,0], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('COVID-19 Learning Curve')
plt.savefig('./C19_plots/C19_train_loss.png')
plt.clf()

# val dice
plt.plot(epoch[:T],C19data[0,:T,4], color='b', linewidth=2.5)
plt.plot(epoch[:T],C19data[1,:T,4], color='g', linewidth=2.5)
plt.plot(epoch[:T],C19data[2,:T,4], color='m', linewidth=2.5)
plt.plot(epoch[:T],C19data[3,:T,4], color='c', linewidth=2.5)
plt.plot(epoch[:T],C19data[4,:T,4], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Validation Dice Coefficient')
plt.title('COVID-19 Validation Dice Coefficient')
plt.savefig('./C19_plots/C19_val_dice.png')
plt.clf()

#train dice
plt.plot(epoch[:T],C19data[0,:T,1], color='b', linewidth=2.5)
plt.plot(epoch[:T],C19data[1,:T,1], color='g', linewidth=2.5)
plt.plot(epoch[:T],C19data[2,:T,1], color='m', linewidth=2.5)
plt.plot(epoch[:T],C19data[3,:T,1], color='c', linewidth=2.5)
plt.plot(epoch[:T],C19data[4,:T,1], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Training Dice Coefficient')
plt.title('COVID-19 Training Dice Coefficient')
plt.savefig('./C19_plots/C19_train_dice.png')
plt.clf()

#val iou
plt.plot(epoch[:T],C19data[0,:T,5], color='b', linewidth=2.5)
plt.plot(epoch[:T],C19data[1,:T,5], color='g', linewidth=2.5)
plt.plot(epoch[:T],C19data[2,:T,5], color='m', linewidth=2.5)
plt.plot(epoch[:T],C19data[3,:T,5], color='c', linewidth=2.5)
plt.plot(epoch[:T],C19data[4,:T,5], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Validation IoU')
plt.title('COVID-19 Validation IoU')
plt.savefig('./C19_plots/C19_val_iou.png')
plt.clf()

#train iou
plt.plot(epoch[:T],C19data[0,:T,2], color='b', linewidth=2.5)
plt.plot(epoch[:T],C19data[1,:T,2], color='g', linewidth=2.5)
plt.plot(epoch[:T],C19data[2,:T,2], color='m', linewidth=2.5)
plt.plot(epoch[:T],C19data[3,:T,2], color='c', linewidth=2.5)
plt.plot(epoch[:T],C19data[4,:T,2], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Training IoU')
plt.title('COVID-19 Training IoU')
plt.savefig('./C19_plots/C19_train_iou.png')
plt.clf()


############### Lung CT data

T = 30
# val loss
plt.plot(epoch[:T],LCTdata[0,:T,3], color='b', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[1,:T,3], color='g', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[2,:T,3], color='m', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[3,:T,3], color='c', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[4,:T,3], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('Lung CT data Learning Curve')
plt.savefig('./LCT_plots/LCT_val_loss.png')
plt.clf()

# train loss
plt.plot(epoch[:T],LCTdata[0,:T,0], color='b', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[1,:T,0], color='g', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[2,:T,0], color='m', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[3,:T,0], color='c', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[4,:T,0], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Lung CT data Learning Curve')
plt.savefig('./LCT_plots/LCT_train_loss.png')
plt.clf()

# val dice
plt.plot(epoch[:T],LCTdata[0,:T,4], color='b', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[1,:T,4], color='g', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[2,:T,4], color='m', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[3,:T,4], color='c', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[4,:T,4], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Validation Dice Coefficient')
plt.title('Lung CT data Validation Dice Coefficient')
plt.savefig('./LCT_plots/LCT_val_dice.png')
plt.clf()

#train dice
plt.plot(epoch[:T],LCTdata[0,:T,1], color='b', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[1,:T,1], color='g', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[2,:T,1], color='m', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[3,:T,1], color='c', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[4,:T,1], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Training Dice Coefficient')
plt.title('Lung CT data Training Dice Coefficient')
plt.savefig('./LCT_plots/LCT_train_dice.png')
plt.clf()

#val iou
plt.plot(epoch[:T],LCTdata[0,:T,5], color='b', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[1,:T,5], color='g', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[2,:T,5], color='m', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[3,:T,5], color='c', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[4,:T,5], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Validation IoU')
plt.title('Lung CT data Validation IoU')
plt.savefig('./LCT_plots/LCT_val_iou.png')
plt.clf()

#train iou
plt.plot(epoch[:T],LCTdata[0,:T,2], color='b', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[1,:T,2], color='g', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[2,:T,2], color='m', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[3,:T,2], color='c', linewidth=2.5)
plt.plot(epoch[:T],LCTdata[4,:T,2], color='r', linewidth=2.5)
plt.legend(models)
plt.xlabel('Epochs')
plt.ylabel('Training IoU')
plt.title('Lung CT data Training IoU')
plt.savefig('./LCT_plots/LCT_train_iou.png')
plt.clf()
