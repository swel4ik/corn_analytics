import os
from pathlib import Path
import glob
import shutil

dirs = glob.glob('D:/Samples-png/Samples/valid/*')
for dir in dirs:
    images = os.listdir(dir)
    for image in images:
        shutil.move(f'{dir}/{image}', f'D:/Samples-png/Samples/train/{image}')