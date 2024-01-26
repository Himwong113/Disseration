from pathlib import Path
import torch
import os
import pandas as pd

class Para:
    def __init__(self):
        self.Epoch = 100
        self.ClassCount = 8
        self.batchsize= 8
        self.Classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()
        self.datasource = "pointcloud"
        self.dataset = 'Kittidet'
        self.current_epoch = 0

        self.model_folder = f'{self.dataset}'
        self.model_filename = f'{self.dataset}_{str(self.current_epoch)}.pt'
        self.config_para_log = f"{str(Path('.'))}/{self.model_folder}/config_{self.dataset}_para.csv"
        self.dataset_datapath='/Users/wonglaihim/Masternote/Sem2/Disseration/downsampled/'
        self.device="mps"



    def get_weights_file_path(self):
        return str(Path('.') / self.model_folder / self.model_filename)

    def get_parameters(self):
        attributes = {attr: [getattr(self, attr) ]for attr in vars(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
        return attributes
    

    def para_update(self):
        csv_file_path = self.config_para_log
        parameter=self.get_parameters()
        if os.path.isfile(csv_file_path):
            print(f"CSV file '{self.model_folder}' exists.")
            
            df_update=pd.DataFrame(parameter)
            df_prev= pd.read_csv(csv_file_path)
            result = pd.concat([df_prev, df_update], ignore_index=True)
            result.to_csv(csv_file_path)
            
        else:
            print(f"CSV file '{self.model_folder}' does not exist.")
            df_update=pd.DataFrame(parameter)
           
            result = pd.concat([ df_update], ignore_index=True)
            result.to_csv(csv_file_path,index=False)

"""
config = Para()
print(config.current_epoch)
#print(config.get_weights_file_path())
print(config.para_update())
"""

