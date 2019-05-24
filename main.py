"""
Project of Acoustic Scene Classification

Started on 17 Feb 2019

@author: Plumecocq Simon

First CNN
"""

import os
import numpy as np
import pandas as pd
import time
import torch
import gzip
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)
print('Nb of GPU: '+str(torch.cuda.device_count()))

torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())
print(torch.cuda.memory_cached())

import torch.nn as nn
import torch.optim as optim
from torch.utils import data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


main_path = "C:\\Users\\Plumecocq\\Documents\\Python Scripts\\ProjetAcousticSceneClassification"
#main_path = "M:\SA HOME DRIVE\IA\ProjectAudioSceneDetection"
os.chdir(main_path)

#from simpleCNN import SimpleCNN
from simpleCNN_v2 import SimpleCNN
from CNN_Dorfer import CNN_Dorfer
from CNN_Dorfer2 import CNN_Dorfer2
from progressbar import progress_bar
from save_model import save_model


data_path = "data"
data_path = "E:\\ProjetAcousticSceneClassification\\data"
dev_save1_path = "transform_1"
dev_save2_path = "transform_2"
dev_save3_path = "transform_3"
dev_path_save1 = data_path+"\\"+dev_save1_path+"\\"
dev_path_save2 = data_path+"\\"+dev_save2_path+"\\"
dev_path_save3 = data_path+"\\"+dev_save3_path+"\\"

result_path = "result\\cnn_1"

#Build the training dataset
my_file = open(data_path+"\\fold2_train.txt", "r")
files_train = [line[:-1] for line in my_file]
my_file.close()
nb_input = len(files_train)
print('Nb input train: '+str(nb_input))

list_class = ('airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram')
nb_class = len(list_class)
print('Nb class: '+str(nb_class))
#labels = np.zeros((nb_input,nb_class))
labels = np.zeros(nb_input)

for index_file, name_file in enumerate(files_train):
    #if index_file >5000 and index_file < 5010:
    nb_car_class = name_file.find('-')
    index_class = list_class.index(name_file[:nb_car_class])
    #print(name_file+' '+str(nb_car_class)+' '+str(index_class))
    #labels[index_file, index_class] = 1
    labels[index_file] = index_class


#Build the eval dataset
my_file = open(data_path+"\\fold2_evaluate.txt", "r")
files_eval = [line[:-1] for line in my_file]
my_file.close()
nb_eval = len(files_eval)
print('Nb eval: '+str(nb_eval))


#labels_eval = np.zeros((nb_eval,nb_class))
labels_eval = np.zeros(nb_eval) 
for index_file, name_file in enumerate(files_eval):
    nb_car_class = name_file.find('-')
    index_class = list_class.index(name_file[:nb_car_class])
    labels_eval[index_file] = index_class    


class myDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_file, labels, path, mean, std, nb_frame=None):
        'Initialization'
        self.labels = labels
        self.list_file = list_file
        self.path = path
        self.mean = mean
        self.std = std
        self.nb_frame = nb_frame
        if std[0] == 0 or std[1] == 0 or std[2] == 0:
            print('Warning: one standard deviation is at 0')
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_file)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data
        f = gzip.GzipFile(self.path+self.list_file[index][:-4]+".npy.gz", "r")
        data = np.load(f)
        # Potential select of random frames
        if self.nb_frame is not None:
            select = np.random.randint(low=0, high = data.shape[2],size=self.nb_frame)
            select.sort()
            data = data[:,:,select]
        #Create the third dimension L-R
        data_3 = np.zeros((3,data.shape[1],data.shape[2]))
        data_3[0] = (data[0]-self.mean[0])/self.std[0]
        data_3[1] = (data[1]-self.mean[1])/self.std[1]
        data_3[2] = (data[0]-data[1]-self.mean[2])/self.std[2]
        #Transform to Tensor
        X = torch.Tensor(data_3)
        y = torch.Tensor(np.array(self.labels[index]))
        return X, y
        
# Generators
mean_train = np.load("data\\mean_train.npy")
std_train = np.load("data\\std_train.npy")
training_set = myDataset(files_train, labels, dev_path_save3,
                         mean_train, std_train, nb_frame=128)
training_generator = data.DataLoader(training_set, batch_size=32, shuffle=True, pin_memory=True)

eval_set = myDataset(files_eval, labels_eval, dev_path_save3,
                     mean_train, std_train)
eval_generator = data.DataLoader(eval_set, batch_size=32, shuffle=False, pin_memory=True)

#Mix-Up data
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam): 
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b) 

#Build the CNN   
#cnn=SimpleCNN()
#cnn=CNN_Dorfer(device)
cnn=CNN_Dorfer2(device)
print(cnn)
lr_init = 0.001
optimizer = optim.Adam(cnn.parameters(), lr=lr_init)
criterion = nn.CrossEntropyLoss()


def linear_adjust_learning_rate(epoch, epoch_min=25, epoch_max=100): 
    """linear decrease of the learning rate from epoch_min to epoch_max""" 
    if epoch > epoch_min:
        for param_group in optimizer.param_groups: 
            param_group['lr'] -= lr_init/(epoch_max-epoch_min+1)
    return optimizer.param_groups[0]['lr']

save_result = []
def train(nbEpochs=1):
    best_test_acc = 0 
    train_size = len(training_generator.dataset)    
    test_size = len(eval_generator.dataset) 
    for epoch in range(nbEpochs):  
        train_loss = 0.0
        train_acc = 0.0
        cnn.train()
        for index_batch, (inputs, labels) in enumerate(training_generator):
            inputs, labels = inputs.to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  
            progress_bar(index_batch, len(training_generator))
        train_loss /= train_size
        train_acc /= train_size
        print("Train at Epoch:",epoch," loss:", train_loss, " accuracy:",100.0*train_acc)      
        
        test_loss = 0.0
        test_acc = 0.0
        cnn.eval()
        with torch.no_grad():
            for index_batch, (inputs, labels) in enumerate(eval_generator):
                inputs, labels = inputs.to(device), labels.long().to(device)
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels).sum().item()
                test_loss += loss.item()
                progress_bar(index_batch, len(eval_generator))
        test_loss /= test_size
        test_acc /= test_size
        print("Test at Epoch:",epoch," loss:", test_loss, " accuracy:",100.0*test_acc)
        
        lr = linear_adjust_learning_rate(epoch)
        save_result.append([epoch,lr,train_loss,train_acc,test_loss,test_acc])

        if epoch > 30 and test_acc > best_test_acc:
            best_test_acc = test_acc
            #Add save model            

save_result = []
def train_mixup(nbEpochs=1):
    best_test_acc = 0
    train_size = len(training_generator)  
    train_size = len(training_generator.dataset)  
    test_size = len(eval_generator)  
    test_size = len(eval_generator.dataset) 
    for epoch in range(nbEpochs):  
        train_loss = 0.0
        train_acc = 0.0
        cnn.train()
        for index_batch, (inputs, labels) in enumerate(training_generator):
            inputs, labels = inputs.to(device), labels.long().to(device)
            inputs, lbl_a, lbl_b, lam = mixup_data(inputs, labels, alpha) 
            #?? from torch.autograd import Variable
            #inputs, lbl_a, lbl_b = map(Variable, (inputs, lbl_a, lbl_b)) 
            
            optimizer.zero_grad()
            outputs = cnn(inputs)
            #loss = criterion(outputs, labels)
            loss = mixup_criterion(criterion, outputs, lbl_a, lbl_b, lam)
            _, predicted = torch.max(outputs.data, 1)
            #train_acc += (predicted == labels).sum().item()
            train_acc += (lam*(predicted == lbl_a).sum().item() +
                        (1-lam)*(predicted == lbl_b).sum().item() )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  
            progress_bar(index_batch, len(training_generator))
        train_loss /= train_size
        train_acc /= train_size
        print("Train at Epoch:",epoch," loss:", train_loss, " accuracy:",100.0*train_acc)      
        
        test_loss = 0.0
        test_acc = 0.0
        cnn.eval()
        with torch.no_grad():
            for index_batch, (inputs, labels) in enumerate(eval_generator):
                inputs, labels = inputs.to(device), labels.long().to(device)
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels).sum().item()
                test_loss += loss.item()
                progress_bar(index_batch, len(eval_generator))
        test_loss /= test_size
        test_acc /= test_size
        print("Test at Epoch:",epoch," loss:", test_loss, " accuracy:",100.0*test_acc)
        
        lr = linear_adjust_learning_rate(epoch)
        save_result.append([epoch,lr,train_loss,train_acc,test_loss,test_acc])

        if epoch > 30 and test_acc > best_test_acc:
            best_test_acc = test_acc
            #Add save model    
            save_model("Dorfer2",str(epoch))
            
            
#Param for mix-up data
alpha = 0.2           

cnn.to(device)

train(1)
train(20)
train(30)

train_mixup(1)
train_mixup(100)

#Save the result in numpy
ext = 'cnn_Dorfer_withGN_MixUp2_100epochs'
np.save(result_path + "\\save_result_"+ext+".npy",save_result)
#Save the result in csv
save_result_df = pd.DataFrame(save_result,
    columns=['Epoch','Learning rate','Train Loss','Train Accuracy','Test Loss','Test Accuracy'])
save_result_df.to_csv(result_path + "\\save_result_"+ext+".csv", sep=';')


