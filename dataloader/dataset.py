# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:08:02 2022

@author: TAY
"""

import os
from sklearn.model_selection import train_test_split
# from mypath import Path
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import csv

# preprocess=False    处理数据时为True,训练时为False.
class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 
            # Save preprocess data into output_dir
            output_dir = 
            return root_dir, output_dir
        
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = 
            output_dir = 
            return root_dir, output_dir
        
        elif database == 'kinetic100':
            # folder that contains class labels
            root_dir = 
            output_dir = 
            return root_dir, output_dir
        
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'E:/3D/pytorch-video-recognition-master/pytorch-video-recognition-master/ucf101-caffe.pth'


class VideoDataset(Dataset):

    def __init__(self, dataset='hmdb51', split='train', clip_len=8, labeldata = "YES", percent = 0.5):#      处理数据时为True,训练时为False.
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        print(self.root_dir, self.output_dir)
        
        rgbfolder = os.path.join(self.output_dir, 'train(RGB)')
        htgfolder = os.path.join(self.output_dir, 'train(HTG)')
        rgbtestfolder = os.path.join(self.output_dir, 'test(RGB)')
        htgtestfolder = os.path.join(self.output_dir, 'test(HTG)')
        
        self.clip_len = clip_len          #16
        self.split = split
        print(self.split)
        self.labeldata = labeldata
        self.percent = percent
        
        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')
            
        self.rgbnames, self.htgnames, labels = [], [], []
        
        for label in sorted(os.listdir(rgbfolder)):
            
            if self.split == 'train' and self.labeldata == "YES":
                for fname in os.listdir(os.path.join(rgbfolder, label))[:int(len(os.listdir(os.path.join(rgbfolder, label)))*self.percent)]:   # fname：每一个动作文件下的视频名称
                    self.rgbnames.append(os.path.join(rgbfolder, label, fname))
                    labels.append(label)
                for fname in os.listdir(os.path.join(htgfolder, label))[:int(len(os.listdir(os.path.join(htgfolder, label)))*self.percent)]:   # fname：每一个动作文件下的视频名称
                    self.htgnames.append(os.path.join(htgfolder, label, fname))
                    
            elif self.split == 'train' and self.labeldata == "NO":
                for fname in os.listdir(os.path.join(rgbfolder, label))[int(len(os.listdir(os.path.join(rgbfolder, label)))*self.percent):]:   # fname：每一个动作文件下的视频名称
                    self.rgbnames.append(os.path.join(rgbfolder, label, fname))
                    labels.append(label)
                for fname in os.listdir(os.path.join(htgfolder, label))[int(len(os.listdir(os.path.join(htgfolder, label)))*self.percent):]:   # fname：每一个动作文件下的视频名称
                    self.htgnames.append(os.path.join(htgfolder, label, fname))
                    
            elif self.split == 'test':
                for fname in os.listdir(os.path.join(rgbtestfolder, label)):   # fname：每一个动作文件下的视频名称
                    self.rgbnames.append(os.path.join(rgbtestfolder, label, fname))
                    labels.append(label)
                for fname in os.listdir(os.path.join(htgtestfolder, label)):   # fname：每一个动作文件下的视频名称
                    self.htgnames.append(os.path.join(htgtestfolder, label, fname))
                    
        assert len(labels) == len(self.rgbnames) == len(self.htgnames)
        print('Number of {} videos: {:d}'.format(split, len(self.rgbnames)))

        # Prepare a mapping between the label names (strings) and indices (ints) {'brush_hair': 0, 'cartwheel': 1, 'catch': 2}
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        
    def __len__(self):
        return len(self.rgbnames)
    
    def __getitem__(self, index):
        # Loading and preprocessing.
        bufferrgb = self.load_frames(self.rgbnames[index]) # 先加载连续帧的数据
        bufferhtg = self.load_frames(self.htgnames[index]) # 先加载连续帧的数据
        bufferrgb = self.crop(bufferrgb, self.clip_len, self.crop_size)
        bufferhtg = self.crop(bufferhtg, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])
        
        if self.split == 'train':
            # Perform data augmentation
            bufferrgb = self.randomflip(bufferrgb)
            bufferhtg = self.randomflip(bufferhtg)
        bufferrgb = self.normalize(bufferrgb)
        bufferrgb = self.to_tensor(bufferrgb)
        bufferhtg = self.normalize(bufferhtg)
        bufferhtg = self.to_tensor(bufferhtg)
        
        return torch.from_numpy(bufferrgb), torch.from_numpy(bufferhtg), torch.from_numpy(labels)
    
    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True
        
    def random_crop(image):
        min_ratio = 0.5
        max_ratio = 1
    
        w, h = image.size
    
        ratio = random.random()
    
        scale = min_ratio + ratio * (max_ratio - min_ratio)
    
        new_h = int(h * scale)
        new_w = int(w * scale)
    
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)
    
        image = image.crop((x, y, x + new_w, y + new_h))
    
        return image
    
    def randomflip(self, buffer):
        
        if np.random.random() < 0.5:
            if np.random.random() < 0.25:
                for i, frame in enumerate(buffer):
                    frame = cv2.flip(buffer[i], flipCode=1) # 图像以0.5的概率进行反转
                    buffer[i] = cv2.flip(frame, flipCode=1)
            else:
                for i, frame in enumerate(buffer):
                    frame = random_crop(frame)
                    buffer[i] = frame
                
        return buffer

    def normalize(self, buffer):
        
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
            
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)]) # 排序
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len) # 返回(0, buffer.shape[0] - clip_len)之间的数

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size, :]
        return buffer

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='hmdb51', split='train', clip_len=8, labeldata = "NO", percent = 0.2)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
    test_data = VideoDataset(dataset='hmdb51', split='test', clip_len=8)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=4)
    
    print(len(train_data))
    print(len(train_loader))
    
    for index, (rgb, htg, labels) in enumerate(test_loader):
        print(rgb.shape)
        print(htg.shape)
        
        if index == 0:
            break
        

