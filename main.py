import numpy as np
import os
import shutil

if __name__ == '__main__':
    txt_dir = '/fru92_images/fru_train_0.txt'
    data_txt = open(txt_dir, 'r')
    imgs = []
    for line in data_txt:
        line = line.strip()
        class_name = line.split('/')[0]
        folder_path = '/ai09/fru92_k0/train/' + class_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        words = line.split(' ')
        imgs.append((" ".join(words[:-1]), int(words[-1])))
        source_path = '/fru92_images/' + imgs[0][0]
        destination_path = '/ai09/fru92_k0/train/' + imgs[0][0]
