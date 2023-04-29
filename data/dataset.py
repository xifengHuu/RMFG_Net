import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from config.config import config
from utils.image_utils import *


class PreTrainDataset(Dataset):
    def __init__(self, train_path, transform=None):
        super(Dataset, self).__init__()
        self.file_list = []
        for dataset in train_path:
            data_dir = config.train_dataset_root + dataset
            for dir in os.listdir(data_dir):
                image_dir = os.path.join(data_dir, dir)
                gt_path = image_dir + '/GT/'
                img_path = image_dir + '/Frame/'
                img_list = []
                for name in os.listdir(img_path):
                    if not name.startswith('.'):
                        img_list.append(name)
                self.file_list.extend([(img_path + name,
                                        gt_path + name.replace('jpg', 'png'))
                                    for name in img_list])

        self.img_label_transform = transform
    
    def __getitem__(self, idx):
        img_path, label_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        img, label = self._process(img, label)
        return img, label, img_path

    def _process(self, img, label):
        if self.img_label_transform:
            img, label = self.img_label_transform(img, label)
        return img, label

    def __len__(self):
        return len(self.file_list)


class FinetuneDataset(Dataset):
    def __init__(self, finetune_path, transform=None, time_interval=1):
        super(FinetuneDataset, self).__init__()
        self.time_clips = config.finetune_time_clips
        self.video_list = []

        for video_name in finetune_path:
            video_root = os.path.join(config.finetune_dataset_root, video_name)
            img_path = os.path.join(video_root, "Frame")
            label_path = os.path.join(video_root, "GT")
            img_list = os.listdir(img_path)
            img_list.sort(key=lambda x:int(x.split('.')[0]))

            self.video_filelist = []
            for img_name in img_list:
                if img_name.startswith('.'):
                    continue    
                self.video_filelist.append((
                    os.path.join(img_path, img_name),
                    os.path.join(label_path, img_name)
                ))

            for begin in range(0, len(self.video_filelist) - (self.time_clips - 1) * time_interval - 1):
                batch_clips = []
                for t in range(self.time_clips):
                    batch_clips.append(self.video_filelist[begin + time_interval * t])
                self.video_list.append(batch_clips)

        self.img_label_transform = transform
    
    def __getitem__(self, idx):
        img_label_pair = self.video_list[idx]
        img_list, label_list, img_all_path = [], [], []
        for idx, (img_path, label_path) in enumerate(img_label_pair):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_list.append(img)
            label_list.append(label)
            img_all_path.append(img_path)
        img_list, label_list = self._process(img_list, label_list)
        inputs, labels = torch.zeros(len(img_list), *(img_list[0].shape)), torch.zeros(len(label_list), *(label_list[0].shape))
        for idx, (img, label, img_path) in enumerate(zip(img_list, label_list, img_all_path)):
            inputs[idx, :, :, :] = img
            labels[idx, :, :, :] = label    
        return inputs, labels

    def _process(self, img_list, label_list):
        if self.img_label_transform:
            img_list, label_list = self.img_label_transform(img_list, label_list)
        return img_list, label_list
    
    def __len__(self):
        return len(self.video_list)

class TestDataset(Dataset):
    def __init__(self, root, dataset_part, transform):
        time_interval = 1
        self.video_filelist = dataset_part
        self.time_clips = config.test_time_clips
        self.video_test_list = []

        video_root = os.path.join(root, dataset_part, 'Frame')
        img_list = os.listdir(video_root)
        self.video_filelist = []
        img_list.sort(key=lambda x:int(x.split('.')[0]))
        for filename in img_list:
            self.video_filelist.append(os.path.join(video_root, filename))

        begin = 0
        while begin < len(self.video_filelist) - 1:
            if len(self.video_filelist) - 1 - begin <= self.time_clips:
                begin = len(self.video_filelist) - self.time_clips
            batch_clips = []
            for t in range(self.time_clips):
                batch_clips.append(self.video_filelist[begin + time_interval * t])
            begin += self.time_clips
            self.video_test_list.append(batch_clips)

        self.img_transform = transform

    def __getitem__(self, idx):
        img_path_list = self.video_test_list[idx]
        img_list = []
        for idx, img_path in enumerate(img_path_list):
            img = Image.open(img_path).convert('RGB')
            img_list.append(self.img_transform(img))
        inputs = torch.zeros(len(img_list), *(img_list[0].shape))
        for idx, img in enumerate(img_list):
            inputs[idx, :, :, :] = img
        return inputs, img_path_list

    def __len__(self):
        return len(self.video_test_list)


def get_train_dataset(local_rank=0):
    train_transform = PretrainComposeTensor([
        PretrainResize(config.size[0], config.size[1]), 
        PretrainRandomCropResize(15),
        PretrainRandomHorizontalFlip(0.5),
        PretrainToTensor(),
        PretrainNormalize(config.train_mean_std[0], config.train_mean_std[1])
    ])
    print("get_train_dataset", config.train_mean_std)
    train_dataset = PreTrainDataset(config.train_dataset_list, transform=train_transform)
    return train_dataset


def get_finetune_dataset(local_rank=0):
    finetune_transform = PretrainComposeTensor([
        FinetuneResize(config.size[0], config.size[1]),
        FinetuneRandomCropResize(7),
        FinetuneRandomHorizontalFlip(0.5),
        FinetuneToTensor(),
        FinetuneNormalize(config.finetune_mean_std[0], config.finetune_mean_std[1])
    ])
    print("get_finetune_dataset", config.finetune_mean_std)
    finetune_dataset = FinetuneDataset(config.finetune_dataset_list, transform=finetune_transform, time_interval=1)
    return finetune_dataset

def get_test_dataset(dataset_part=""):
    test_transform = TestComposeTensor([
        TestResize(config.size[0], config.size[1]),
        TestToTensor(),
        TestNormalize(config.test_mean_std[0], config.test_mean_std[1])
    ])
    print(dataset_part, " get_test_dataset: ", config.test_mean_std)
    test_dataset = TestDataset(config.test_dataset_root, dataset_part, transform=test_transform)
    return test_dataset