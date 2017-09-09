import cv2
import os
import logging as log
import datetime as dt
import numpy as np
from time import sleep
from icrawler.builtin import GoogleImageCrawler
import tensorlayer as tl
import matplotlib.pyplot as plt

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
'''
keyward_list = ['新垣 結衣', '長澤まさみ', '石原さとみ', '綾瀬はるか', '堀北真希', '上戸彩', '戸田恵梨香']
keyward_eng = ['gakki', 'masami', 'satomi', 'haruka', 'maki', 'aya', 'toda']
keyward_list = ['有村架純', '広瀬すず', '深田恭子', '廣末涼子', '佐々木希', '北川景子', '桐谷美玲 ']
keyward_eng = ['mura', 'suzu', 'kyoko', 'ryoko', 'nozomi', 'keiko', 'mirei']
'''

keyward_list = ['橋本環奈', '土屋太鳳', '松岡茉優', '波瑠', '吉高 由里子', '満島ひかり', '榮倉奈々']
keyward_eng = ['kanna', 'tao', 'mayu', 'haru', 'yuriko', 'hikari', 'nana']

#path = 'C:\\Users\\233\\Downloads'
save_dir = 'actress'
img_size = 72
crop_size = 64

def image_downloader(key_ward, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': save_dir})
    google_crawler.crawl(keyword=key_ward, max_num=10000000, date_min=None, 
                         date_max=None, min_size=(200, 200), max_size=None)


def process(image, save_dir, augment = False):
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, 
                                         minNeighbors = 3, minSize = (30, 30))
    '''
    for (x, y, w, h) in faces:
        
        x_offset, y_offset = int(0.1 * w), int(0.1 * h)
        crop = image[y-y_offset: y+h+y_offset, x-x_offset: x+w+x_offset].copy()
    '''
    if faces is not ():
        (x, y, w, h) = faces[0]
        if augment == True:
            x_start, y_start = int(x- 0.1 * w), int(y - 0.1 * h)
            x_offset, y_offset = int(1.2 * w), int(1.2 * h)
            img_size = 72
        else:
            x_start, y_start = x, y 
            x_offset, y_offset = w, h
            img_size = 64
            
        crop = image[y_start: y_start + y_offset, x_start: x_start + x_offset].copy()
        if crop.any():
            crop = cv2.resize(crop, (img_size, img_size))
            cv2.imwrite(save_dir, crop)
        

def prepare_batch_image(keyward_list, keyward_eng, save_dir):
    for i in range(len(keyward_list)):
        img_dir = os.path.join(save_dir, keyward_eng[i])
        image_downloader(keyward_list[i], img_dir)
        j = 0
        for filename in os.listdir(img_dir):
            img = cv2.imread(os.path.join(img_dir,filename))
            save_crop_dir = os.path.join(save_dir, keyward_eng[i]+str(j)+'.jpg')
            if img is not None:
                process(img, save_crop_dir, augment = True)
            j += 1
            
            
def load_image(save_dir):
    image = []
    #load_dir = os.path.join(path, train_dir)
    for filename in os.listdir(save_dir):
        img = cv2.imread(os.path.join(save_dir, filename))
        image.append(img)
    return image


def rand_crop(image, origin_size, crop_size):
    offset_w = np.random.randint(0, origin_size - crop_size)
    offset_h = np.random.randint(0, origin_size - crop_size)
    crop_image = image[offset_h: offset_h + crop_size, offset_w: offset_w + crop_size]
    return crop_image


def rand_flip(image):
    flip_prop = np.random.randint(0,20)
    if flip_prop > 9:
        image = cv2.flip(image, 1)
    return image


def next_batch(batch_size, data):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    for j in range(len(data_shuffle)):
        data_shuffle[j] = rand_crop(data_shuffle[j], img_size, crop_size)
        data_shuffle[j] = rand_flip(data_shuffle[j])
        data_shuffle[j] = np.array(data_shuffle[j])/127.5 - 1
    data_shuffle = np.asarray(data_shuffle)
    return data_shuffle


def print_image(image, save_dir):
    tl.files.exists_or_mkdir(save_dir)
    for j in range(len(image)):
        #img[j] = (img[j] + 1) * 127.5
        img_dir = os.path.join(save_dir, str(j)+'.jpg')
        image[j] = (image[j] + 1) * 127.5
        cv2.imwrite(img_dir, image[j])


def print_image(image, save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fused_dir = os.path.join(save_dir, 'fused_image.jpg')
    fused_image = [0] * 8
    for i in range(8):
        fused_image[i] = []
        for j in range(8):
            k = i * 8 + j
            image[k] = (image[k] + 1) * 127.5
            fused_image[i].append(image[k])
            img_dir = os.path.join(save_dir, name+str(k)+'.jpg')
            cv2.imwrite(img_dir, image[k])
        fused_image[i] = np.hstack(fused_image[i])
        #fused_image[i] = np.concatenate(fused_image[i], axis = 1)
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image)
    

'''
def main():
    prepare_batch_image(keyward_list, keyward_eng, save_dir)
    
if __name__ == '__main__':
    main()
'''
'''
def save_batch(batch_size, data, save_dir):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    for j in range(len(data_shuffle)):
        data_shuffle[j] = rand_crop(data_shuffle[j], img_size, crop_size)
        data_shuffle[j] = rand_flip(data_shuffle[j])
        data_shuffle[j] = np.array(data_shuffle[j])/127.5 - 1
        #data_shuffle[j] = np.asarray(data_shuffle[j])
        cv2.imwrite(save_dir, data_shuffle[j])
'''

augment_dir = 'augment'
if not os.path.exists(augment_dir):
    os.makedirs(augment_dir)
images = load_image(save_dir)
for i in range(1000):
    batch = next_batch(64, images)
    name = 'no%d_'%i
    print_image(batch, augment_dir, name)
'''
for i in range(1000):
    idx = np.arange(0 , len(augment_dir))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [images[i] for i in idx]
    for j in range(len(data_shuffle)):
        data_shuffle[j] = rand_crop(data_shuffle[j], img_size, crop_size)
        data_shuffle[j] = rand_flip(data_shuffle[j])
        data_shuffle[j] = np.array(data_shuffle[j])/127.5 - 1
        data_shuffle[j] = np.asarray(data_shuffle[j])
        cv2.imwrite(augment_dir, data_shuffle[j])
'''


