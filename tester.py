
import numpy as np
import random as rand

arr = [[rand.randint(1,10) for i in range(10)], [rand.randint(1,10) for i in range(10)], [rand.randint(1,10) for i in range(10)]]
arr = np.array(arr)
print(f'rand array : {arr}')
max= np.max(arr)
min= np.min(arr)
norm_arr = np.floor(100*((arr-min)/(max-min))).astype(int)
print(f'normlize array : {norm_arr}')
