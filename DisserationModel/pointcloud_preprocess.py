import torch
import DisserationConfig as Configure
import numpy as np
import tqdm 
from torch.utils.data import  TensorDataset
config= Configure.Para()

def pcNormalization(arr, top):
    arr = np.array(arr)
    max= np.max(arr)
    min= np.min(arr)
    norm_arr = np.floor(top*((arr-min)/(max-min))).astype(int)
    return norm_arr




def create_dataloader(train_dataset,test_dataset):

    


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batchsize,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset= test_dataset,
                                               batch_size=config.batchsize,
                                               shuffle=True)
    
    
    
    return train_loader,test_loader

if __name__ == '__main__':
    print("")


"""


def create_dataloader(train_dataset,test_dataset,trainlabel,testlabel):

    #Normalize
    topcorner = 100
    train_dataset_arr=[]
    test_dataset_arr=[]
    train_label_arr=[]
    test_label_arr=[]

    pbar =tqdm.tqdm(len(train_dataset))
    print("\nProcessing train_dataloder\n")
    for i in range(len(train_dataset)):
        train_dataset_arr.append( pcNormalization(train_dataset[i],topcorner))
        train_label_arr.append(trainlabel[i])
        pbar.update(1)
    print("\n train_dataloder DONE \n")
    pbar =tqdm.tqdm(len(test_dataset))
    print("Processing val_dataloder")
    for i in range (len (test_dataset)):
        test_dataset_arr.append( pcNormalization(test_dataset[i],topcorner))
        test_label_arr.append(testlabel[i])
        pbar.update(1)
    print("\n val_dataloder DONE \n")

    train_concat_dataset =TensorDataset(np.array(train_dataset_arr),np.array(train_label_arr))
    test_concat_dataset =TensorDataset(np.array(test_dataset_arr),np.array(test_label_arr))

    train_loader = torch.utils.data.DataLoader(dataset=train_concat_dataset,
                                               batch_size=config.batchsize,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset= test_concat_dataset,
                                               batch_size=config.batchsize,
                                               shuffle=True)
    

"""

"""
config = Para()
filepath = config.dataset_datapath

sid =0
pointcloud=np.load(filepath+str(sid).zfill(6))
print(pointcloud)
"""
