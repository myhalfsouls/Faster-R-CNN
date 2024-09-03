import albumentations
import numpy as np
from PIL import Image
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

"""
corresponding bounding box scaling is taken from: https://sheldonsebastian94.medium.com/resizing-image-and-bounding-boxes-for-object-detection-7b9d9463125a
"""


class MRCDataset(Dataset):
    def __init__(self, dataset, img_size, str2id, dataset_type):
        """
        :type dataset: fiftyone.core.dataset.Dataset
        """

        self.dataset_type = dataset_type
        self.n_samples = None
        self.images = None
        self.labels = None
        self.bboxes = None
        self.parse_dataset(dataset, img_size, str2id)

    def parse_dataset(self, dataset, img_size, str2id):
        # create resize transform pipeline
        h_out, w_out = img_size
        transform = albumentations.Compose([albumentations.Resize(height=h_out, width=w_out, always_apply=True)],
                                           bbox_params=albumentations.BboxParams(format='pascal_voc'))

        images, labels, bboxes, masks = [], [], [], []
        for image_id, file_path in enumerate(tqdm(dataset.values("filepath"), desc="Pre-processing [{}] Dataset".format(self.dataset_type))):
            # load the image data and convert it to a tensor
            pil_img = Image.open(file_path).convert("RGB")
            img_data = pil_to_tensor(pil_img).type(torch.float32)
            _, h_img, w_img = img_data.shape

            # load the sample
            sample = dataset[file_path]

            # parse out the detections
            max_mask_dim1 = 0
            max_mask_dim2 = 0
            
            labels_i, bboxes_i, masks_i = [], [], []
            for detection in sample["ground_truth"].detections:
                x_min, y_min, w, h = detection.bounding_box
                x_min, w = x_min * w_img, w * w_img
                y_min, h = y_min * h_img, h * h_img
                x_max, y_max = x_min + w, y_min + h
                label = str2id[detection.label]
                labels_i.append(label)
                bboxes_i.append([x_min, y_min, x_max, y_max, label])
                masks_i.append(torch.tensor(detection.mask))
                
                max_mask_dim1 = max(detection.mask.shape[0], max_mask_dim1) 
                max_mask_dim2 = max(detection.mask.shape[1], max_mask_dim2) 

            # perform transformations for re-sizing
            transformed = transform(image=img_data[0, :].cpu().numpy(), bboxes=np.array(bboxes_i))
            img_data = pil_to_tensor(pil_img.resize(img_size[::-1])).type(torch.float32).cpu()

            images.append(img_data)
            labels.append(torch.tensor(labels_i))
            tboxes = torch.tensor(transformed['bboxes'])[:, :-1]
            bboxes.append(tboxes)
            masks.append(masks_i)            

        # store the dataset information
        self.n_samples = len(labels)
        self.images = torch.stack(images).cpu()
        self.labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        self.bboxes = pad_sequence(bboxes, batch_first=True, padding_value=-1)
        self.masks = self.pad_masks(masks, max_mask_dim1, max_mask_dim2)
        self.masks = pad_sequence(self.masks, batch_first=True, padding_value=-1)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.images[index].clone(), self.labels[index].clone(), self.bboxes[index].clone(), self.masks[index]
    
    def pad_masks(self, img_masks, max_dim1, max_dim2):
        padded = []
        for masks in img_masks:
            padded_masks = torch.stack([
                F.pad(mask, (0, max_dim2 - mask.shape[1], 0, max_dim1 - mask.shape[0]), mode='constant', value=-1)
                for mask in masks
            ])
            padded.append(padded_masks)
            
        return padded