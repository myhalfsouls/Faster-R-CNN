import fiftyone.utils.coco as fouc
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import os
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from utils.FRCDataset import FRCDataset
from utils.MRCDataset import MRCDataset


"""
This utility is adapted from: https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb
"""


def load_data(dataset_name, num_train, num_val, img_size, load_masks=False):
    """
    :type dataset_name: str
    :type num_train: int | None
    :type num_val: int | None
    :type img_size: tuple
    :type load_masks: bool
    :return: train_data, val_data, str2id, id2str
    """

    # load in the raw datasets from zoo
    dataset_train = foz.load_zoo_dataset(
        dataset_name,
        split="train",
        max_samples=num_train,
        label_types=["segmentations"]
    )
    dataset_val = foz.load_zoo_dataset(
        dataset_name,
        split="validation",
        max_samples=num_val,
        label_types=["segmentations"]
    )

    # load in the id-to-string mapping and vice-versa
    id2str = load_id2str_mapping(dataset_name)
    id2str[-1] = "pad"
    str2id = {id2str[int_id]: int_id for int_id in id2str.keys()}

    dataset = MRCDataset if load_masks else FRCDataset

    # parse the datasets
    train_data = dataset(dataset_train, img_size, str2id, 'Train')
    val_data = dataset(dataset_val, img_size, str2id, 'Validation')
    return train_data, val_data, str2id, id2str


def load_supplemented_data(dataset_name, num_train, num_val, img_size, load_masks=False):
    """
    :type dataset_name: str
    :type num_train: int | None
    :type num_val: int | None
    :type img_size: tuple
    :type load_masks: bool
    :return: train_data, val_data, str2id, id2str
    """

    fo.config.default_ml_backend = "tensorflow"

    # load in the raw datasets from zoo
    dataset_train = foz.load_zoo_dataset(
        dataset_name,
        split="train",
        max_samples=num_train,
        label_types=["segmentations"]
    )
    if num_train is None:
        num_train = len(dataset_train.values("filepath"))
    dataset_test = foz.load_zoo_dataset(
        dataset_name,
        split="test",
        max_samples=num_train,
        label_types=["segmentations"]
    )
    print("Supplementing {} training images with {} additional images from 'test' split".format(num_train, num_val, dataset_name))
    dataset_val = foz.load_zoo_dataset(
        dataset_name,
        split="validation",
        max_samples=num_val,
        label_types=["segmentations"]
    )

    # load in the id-to-string mapping and vice-versa
    id2str = load_id2str_mapping(dataset_name)
    id2str[-1] = "pad"
    str2id = {id2str[int_id]: int_id for int_id in id2str.keys()}

    dataset = MRCDataset if load_masks else FRCDataset

    # parse the datasets
    train_data = dataset([dataset_train, dataset_test], img_size, str2id, 'Train')
    val_data = dataset(dataset_val, img_size, str2id, 'Validation')
    return train_data, val_data, str2id, id2str


def load_id2str_mapping(dataset_name):
    filename = os.path.join("config", dataset_name, "id2str_mapping.txt")
    id2str = {}
    with open(filename, 'r') as file:
        lines = file.read().split('\n')
        for line in lines:
            split = line.split('=')
            id2str[int(split[0])] = split[1]
    return id2str


def load_data_old(dataset_train, dataset_val, img_size):
    """
    :type dataset_train: fiftyone.core.dataset.Dataset
    :type dataset_val: fiftyone.core.dataset.Dataset
    :type img_size: tuple
    :return: train_data, val_data, cat_ids
    """

    # parse out the number of unique categories
    cat_ids_train = dataset_train.distinct("ground_truth.detections.label")
    cat_ids_val = dataset_val.distinct("ground_truth.detections.label")
    cat_ids = {label: cat_id for cat_id, label in enumerate(np.unique(cat_ids_train + cat_ids_val))}

    # add an entry for the padding
    cat_ids["pad"] = -1

    # parse the datasets
    train_data = FRCDataset(dataset_train, img_size, cat_ids, 'Train')
    val_data = FRCDataset(dataset_val, img_size, cat_ids, 'Validation')
    return train_data, val_data, cat_ids

    # # parse the datasets
    # x_train, y_train = parse_dataset(dataset_train, cat_ids)
    # x_val, y_val = parse_dataset(dataset_val, cat_ids)
    #
    # return x_train, y_train, x_val, y_val, cat_ids


def parse_dataset(dataset, cat_ids):
    """
    converts dataset into:
        x_data: list of image tensors
        y_data: list of target dictionaries, which each have bounding boxes and labels

    :type dataset: fiftyone.core.dataset.Dataset
    :type cat_ids: dict[str, int]
    :param dataset: input fiftyone dataset
    :param cat_ids: dictionary to map category ids to an integer
    :return: x_data, y_data
    """

    x_data = []
    y_data = []
    for image_id, file_path in enumerate(dataset.values("filepath")):
        x_result, y_result = parse_entry(file_path, dataset[file_path], image_id, cat_ids)
        x_data.append(x_result)
        y_data.append(y_result)

    return x_data, y_data


def parse_entry(file_path, sample, image_id, cat_ids):
    # load the image data and convert it to a tensor
    img_data = pil_to_tensor(Image.open(file_path).convert("RGB")).type(torch.float32)

    # parse out the classification and identification data
    boxes, labels = [], []
    for detection in sample["ground_truth"].detections:
        cat_id = cat_ids[detection.label]
        coco_obj = fouc.COCOObject.from_label(detection, sample.metadata, category_id=cat_id)
        x, y, w, h = coco_obj.bbox
        boxes.append([x, y, x + w, y + h])
        labels.append(coco_obj.category_id)

    target_data = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "id": torch.as_tensor([image_id])
    }

    return img_data, target_data
