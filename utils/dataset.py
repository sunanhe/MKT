import os
import pickle
import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.transforms import build_transform

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict   


def build_dataloader(args):

    val_img_names = load_dict(os.path.join(args.data_path, 'test_img_names.pkl'))
    test_image_filenames = val_img_names['img_names']
    transform = build_transform(False, args)
    val_dataset = ValDataset(args, test_image_filenames, transform)

    img_names = load_dict(os.path.join(args.data_path, 'img_names.pkl'))
    image_filenames = img_names['img_names']
    transform = build_transform(True, args)
    train_image_names = np.array(image_filenames)
    train_dataset = TrainDataset(args, train_image_names, transform)

    return train_dataset, val_dataset

class TrainDataset(Dataset):

    def __init__(self, args, image_names, transforms):
        self.src = args.data_path
        train_loc = os.path.join(self.src, 'features' ,'nus_wide_train.h5')
        self.train_features = h5py.File(train_loc, 'r')
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        
        t = file_name.split("_")
        path = os.path.join(self.src, "Flickr", "_".join(t[:-2]), t[-2]+"_"+t[-1])
        img = Image.open(path).convert('RGB')
        inputs = self.transforms(img)
        label = np.int32(self.train_features.get(file_name+'-labels'))
            
        label = torch.from_numpy(label).long()

        return inputs, label

    def __len__(self):
        return len(self.image_names)

class ValDataset(Dataset):

    def __init__(self, args, image_names, transforms):
        self.src = args.data_path
        train_loc = os.path.join(self.src, 'features' ,'nus_wide_test.h5')
        self.train_features = h5py.File(train_loc, 'r')
        self.image_names = image_names
        self.transforms = transforms
        
    def __getitem__(self, idx):
        file_name = self.image_names[idx]

        t = file_name.split("_")
        path = os.path.join(self.src, "Flickr", "_".join(t[:-2]), t[-2]+"_"+t[-1])
        img = Image.open(path).convert('RGB')
        inputs = self.transforms(img)

        labels_1006 =  np.int32(self.train_features.get(file_name+'-labels'))
        labels_81 =  np.int32(self.train_features.get(file_name+'-labels_81'))

        return inputs, labels_1006, labels_81, file_name

    def __len__(self):
        return len(self.image_names)


def build_inf_dataloader(args):

    transform = build_transform(False, args)
    return  Filelist(args.filelist, args.img_root, transform)

class Filelist(Dataset):

    def __init__(self, filelist, root, transforms):
        
        self.items = []
        self.root = root
        self.transforms = transforms
        filelist = open(filelist).readlines()
        for file in filelist:
            self.items.append(file.strip())

    def __getitem__(self, idx):

        item = self.items[idx]
        label = " ".join(item.split(" ")[1:])
        file_name = os.path.join(self.root, item.split(" ")[0])
        img = Image.open(file_name).convert('RGB')
        inputs = self.transforms(img)

        return inputs, label, file_name

    def __len__(self):
        return len(self.items)