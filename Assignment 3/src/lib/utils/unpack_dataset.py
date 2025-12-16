import numpy as np
import pickle
import cv2
import argparse
import os
from tqdm import tqdm

'''
Copyrite: https://gist.github.com/juliensimon/273bef4c5b4490c687b2f92ee721b546
'''

def extractImagesAndLabels(path, file):
    with open(path+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    images = dict[b'data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict[b'labels']
    return images, labels

def extractCategories(path, file):
    with open(path+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    dict = pickle.load(f)
    return dict['label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_path', type=str, default=None)
    args = parser.parse_args()

    batch_path = args.batch_path
    if not os.path.exists(batch_path):
        raise Exception(f"File not found: {batch_path}")

    IMAGE_SAVE_PATH = './data_images/'
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.mkdir(IMAGE_SAVE_PATH)

    for idx, filename in tqdm(enumerate(os.listdir(batch_path))):
        if 'data_batch' in filename: 
            imgarray, _ = extractImagesAndLabels(batch_path, filename)

            for i in range(0, imgarray.shape[0]):
                saveCifarImage(imgarray[i], IMAGE_SAVE_PATH, "image_"+str(idx)+'_'+(str)(i))
        else:
            continue
