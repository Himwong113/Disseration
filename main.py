import DisserationConfig as Configure
import DisserationModel.pointcloud_preprocess_v2 as preprocess
import Viusal_loader.Dataloading_detcopy as visual
import numpy as np 
import pandas as pd
import os
import tqdm
import torch
#visual.show3dpolt(8)

config=Configure.Para()
filepath = config.dataset_datapath
####test
#sid=0



#dataset to tensor 
script_dir = os.path.dirname(os.path.abspath(__file__))
trainsetid_path = os.path.join(script_dir, 'train.txt')
valsetid_path = os.path.join(script_dir, 'val.txt')

with open(trainsetid_path) as f:
    trainlines = f.readlines()  
    trainsetid= [txt.split('\n')[0]for txt in trainlines]
    trainpathdata = [filepath+sid+"/pc.npy" for sid in trainsetid]
    trainpathlabel = [filepath+sid+"/segm.npy" for sid in trainsetid]
        
with open(valsetid_path) as f:
    vallines = f.readlines()
    valsetid= [txt.split('\n')[0]for txt in vallines]
        

for idx ,item in enumerate(trainsetid):
    print(trainpathdata[idx])
    print(trainpathlabel[idx])

    
    train_loader = preprocess.create_dataloader(f'{trainpathdata[idx]}',f'{trainpathlabel[idx]}')

    pbar = tqdm.tqdm(len(train_loader))
    for batch_point_clouds, batch_labels in train_loader:
        #print("Batch Point Clouds:", batch_point_clouds)
        #print("Batch Labels:", batch_labels)
        batch_point_clouds.to(config.device)
        batch_labels.to(config.device)

        pbar.update(1)
    
    



"""
ointcloud_trainset=[[preprocess.pcNormalization(np.load(filepath+sid+"/pc.npy"),1000) ,np.load(filepath+sid+"/segm.npy")]for sid in trainsetid]


#pointcloud_trainsetlabel=[ np.load(filepath+sid+"//segm.npy") for sid in trainsetid]
"""
"""for i ,item in enumerate(pointcloud_trainsetpoint):
    for i ,point in enumerate(item):
        print(point)
"""


"""



pointcloud_valset=[[preprocess.pcNormalization(np.load(filepath+sid+"/pc.npy"),1000),np.load(filepath+sid+"/segm.npy")] for sid in valsetid]
#pointcloud_valsetlabel=[np.load(filepath+sid+"/segm.npy") for sid in valsetid]

train_loader, val_loader = preprocess.create_dataloader(pointcloud_trainset, 
                                                        pointcloud_valset,
                                                        
                                                        )




print("loader ready")
for batch in train_loader:
    print(batch)

    
"""
"""



"""
"""

pointcloud_trainset= [[pointcloud_trainsetpoint[sid],pointcloud_trainsetlabel[sid]]for sid ,item in enumerate(trainsetid)]
train_loader = torch.utils.data.DataLoader(dataset=pointcloud_trainset,
                                               batch_size=config.batchsize,
                                               shuffle=True)
for batch, (item,label) in train_loader:
    print(f'Batch {batch}:')
    print(f'{item}')

train_loader, val_loader = preprocess.create_dataloader(pointcloud_trainset, 
                                                        pointcloud_valset,
                                                        pointcloud_trainsetlabel,
                                                        pointcloud_valsetlabel
                                                        )

for batch, (item,label) in train_loader:
    print(f'Batch {batch}:')
    print(f'{item}')

"""



#
#config.device ="cuda"# you can select mps/ cuda
#config.para_update()
#device = torch.device(config.device) 



