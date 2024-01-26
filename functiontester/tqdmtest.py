from time import sleep
from tqdm import tqdm, trange

times = 0
progress = tqdm(total=1000)

while times < 1000:
    progress.update(1)
    times += 1
    sleep(0.01)