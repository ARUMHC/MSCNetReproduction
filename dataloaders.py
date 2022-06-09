
import torch
import torchvision
import glob
import os
from PIL import Image
import numpy



class DataLoaderSTARE(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        super(DataLoaderSTARE, self).__init__()
        self.image_files = glob.glob(os.path.join(folder_path,'images','*.png')) # .png
        self.mask_segmentation1_files = []
        self.mask_segmentation2_files = []
        self.mask_centerline_files = []
        self.transform = transform
        for img_path in self.image_files:
             self.mask_segmentation1_files.append(os.path.join(folder_path,'mask_segmentation1', os.path.basename(img_path)))
        for img_path in self.image_files:
             self.mask_segmentation2_files.append(os.path.join(folder_path,'mask_segmentation2', os.path.basename(img_path)))
        for img_path in self.image_files:
             self.mask_centerline_files.append(os.path.join(folder_path,'mask_centerline', os.path.basename(img_path)))

    def __getitem__(self, index):
            image_path = self.image_files[index]
            mask_segmentation1_path = self.mask_segmentation1_files[index]
            mask_segmentation2_path = self.mask_segmentation2_files[index]
            mask_centerline_path = self.mask_centerline_files[index]

            image = Image.open(image_path)
            mask_segmentation1 = Image.open(mask_segmentation1_path)
            mask_segmentation2 = Image.open(mask_segmentation2_path)
            mask_centerline = Image.open(mask_centerline_path)

            resize = torchvision.transforms.Resize(size=(624, 624)) # should be 624x624

            img = resize((torchvision.transforms.functional.equalize(torch.from_numpy(numpy.asarray(image, dtype="uint8")).permute(2, 0, 1)).float()/255)[(1,), :, :])

            mask_seg1 = resize(torch.from_numpy(numpy.asarray(mask_segmentation1)/255).float()[None, :])
            mask_seg2 = resize(1-torch.from_numpy(numpy.asarray(mask_segmentation2)/255).float()[None, :])
            mask_center = resize(torch.from_numpy(numpy.asarray(mask_centerline)/255).float()[None, :])


            return img, mask_seg1, mask_seg2, mask_center

    def __len__(self):
        return len(self.image_files)


class DataLoaderCHASE(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        super(DataLoaderCHASE, self).__init__()
        self.image_files = glob.glob(os.path.join(folder_path,'images','*.png')) # .png
        self.mask_segmentation1_files = []
        self.mask_segmentation2_files = []
        self.mask_centerline_files = []
        self.transform = transform
        for img_path in self.image_files:
             self.mask_segmentation1_files.append(os.path.join(folder_path,'mask_segmentation1', os.path.basename(img_path)))
        for img_path in self.image_files:
             self.mask_segmentation2_files.append(os.path.join(folder_path,'mask_segmentation2', os.path.basename(img_path)))
        for img_path in self.image_files:
             self.mask_centerline_files.append(os.path.join(folder_path,'mask_centerline', os.path.basename(img_path)))

    def __getitem__(self, index):
            image_path = self.image_files[index]
            mask_segmentation1_path = self.mask_segmentation1_files[index]
            mask_segmentation2_path = self.mask_segmentation2_files[index]
            mask_centerline_path = self.mask_centerline_files[index]

            image = Image.open(image_path)
            mask_segmentation1 = Image.open(mask_segmentation1_path)
            mask_segmentation2 = Image.open(mask_segmentation2_path)
            mask_centerline = Image.open(mask_centerline_path)

            resize = torchvision.transforms.Resize(size=(960, 960)) # should be 624x624

            img = resize((torchvision.transforms.functional.equalize(torch.from_numpy(numpy.asarray(image, dtype="uint8")).permute(2, 0, 1)).float()/255)[(1,), :, :])

            mask_seg1 = resize(torch.from_numpy(numpy.asarray(mask_segmentation1)/255).float()[None, :])
            mask_seg2 = resize(1-torch.from_numpy(numpy.asarray(mask_segmentation2)/255).float()[None, :])
            mask_center = resize(torch.from_numpy(numpy.asarray(mask_centerline)/255).float()[None, :])


            return img, mask_seg1, mask_seg2, mask_center

    def __len__(self):
        return len(self.image_files)

class DataLoaderDRIVE(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        super(DataLoaderDRIVE, self).__init__()
        self.image_files = glob.glob(os.path.join(folder_path,'images','*.png')) # .png
        self.mask_segmentation1_files = []
        self.mask_segmentation2_files = []
        self.mask_centerline_files = []
        self.transform = transform
        for img_path in self.image_files:
             self.mask_segmentation1_files.append(os.path.join(folder_path,'mask_segmentation1', os.path.basename(img_path)))
        for img_path in self.image_files:
             self.mask_segmentation2_files.append(os.path.join(folder_path,'mask_segmentation2', os.path.basename(img_path)))
        for img_path in self.image_files:
             self.mask_centerline_files.append(os.path.join(folder_path,'mask_centerline', os.path.basename(img_path)))

    def __getitem__(self, index):
            image_path = self.image_files[index]
            mask_segmentation1_path = self.mask_segmentation1_files[index]
            mask_segmentation2_path = self.mask_segmentation2_files[index]
            mask_centerline_path = self.mask_centerline_files[index]

            image = Image.open(image_path)
            mask_segmentation1 = Image.open(mask_segmentation1_path)
            mask_segmentation2 = Image.open(mask_segmentation2_path)
            mask_centerline = Image.open(mask_centerline_path)

            resize = torchvision.transforms.Resize(size=(576, 576))

            img = resize((torchvision.transforms.functional.equalize(torch.from_numpy(numpy.asarray(image, dtype="uint8")).permute(2, 0, 1)).float()/255)[(1,), :, :])

            mask_seg1 = resize(torch.from_numpy(numpy.asarray(mask_segmentation1)/255).float()[None, :])
            mask_seg2 = resize(1-torch.from_numpy(numpy.asarray(mask_segmentation2)/255).float()[None, :])
            mask_center = resize(torch.from_numpy(numpy.asarray(mask_centerline)/255).float()[None, :])


            return img, mask_seg1, mask_seg2, mask_center

    def __len__(self):
        return len(self.image_files)