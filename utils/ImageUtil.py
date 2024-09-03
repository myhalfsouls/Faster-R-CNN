import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from torchvision import ops
import torch


def build_image(image, bboxes, labels, box_color='y', show=False, filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))

    # permute for matplotlib and add to ax
    image_permute = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(image_permute)

    # convert the bounding boxes back to xywh
    bboxes = ops.box_convert(bboxes, in_fmt='xyxy', out_fmt='xywh')

    # add the bounding boxes and the labels to ax
    for bbox, label in zip(bboxes, labels):
        # only display real labels
        if label == "pad":
            continue

        # display bounding box
        x, y, w, h = bbox.detach().cpu().numpy()
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        # display label
        ax.text(x + 5, y + 14, label, bbox=dict(facecolor='white', alpha=0.5))

    if show:
        fig.show()

    if filename is not None:
        fig.savefig(filename)


def build_grid_images(images, bboxes, labels, box_color='y', show=False, filename=None):
    if (len(images) % 3) != 0:
        raise ValueError("build_grid_images requires that len(images) be a multiple of 3, got {}".format(len(images)))
    num_rows = int(len(images) / 3)
    fig, axes = plt.subplots(num_rows, 3, figsize=(8, 9), layout="constrained")

    for i, (image, bboxes_i, labels_i) in enumerate(zip(images, bboxes, labels)):
        axes_idx = np.unravel_index(i, (num_rows, 3))

        # permute for matplotlib and add to ax
        image_permute = image.permute(1, 2, 0).cpu().numpy()
        axes[axes_idx].imshow(image_permute)
        axes[axes_idx].set_xticks([])
        axes[axes_idx].set_yticks([])
        axes[axes_idx].set_xticklabels([])
        axes[axes_idx].set_yticklabels([])

        # convert the bounding boxes back to xywh
        bboxes_i = ops.box_convert(bboxes_i, in_fmt='xyxy', out_fmt='xywh')

        # add the bounding boxes and the labels to ax
        for bbox, label in zip(bboxes_i, labels_i):
            # only display real labels
            if label == "pad":
                continue

            # display bounding box
            x, y, w, h = bbox.detach().cpu().numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=box_color, facecolor='none')
            axes[axes_idx].add_patch(rect)
            # display label
            axes[axes_idx].text(x + 10, y + 35, label, bbox=dict(facecolor='white', alpha=0.5))

    if show:
        fig.show()

    if filename is not None:
        fig.savefig(filename)


def flip_image(image, bboxes):
    width = image.shape[2]
    image_flip = torch.flip(image, dims=[2])
    bboxes_flip = torch.zeros_like(bboxes)
    bboxes_flip[bboxes == -1] = -1
    box_idx = torch.where(bboxes[:, 0] != -1)[0]
    bboxes_flip[box_idx, 2] = width - bboxes[box_idx, 0]
    bboxes_flip[box_idx, 1] = bboxes[box_idx, 1]
    bboxes_flip[box_idx, 0] = width - bboxes[box_idx, 2]
    bboxes_flip[box_idx, 3] = bboxes[box_idx, 3]
    return image_flip, bboxes_flip

def random_flip_batch(images_batch, bboxes_batch):
    batch_size = images_batch.shape[0]
    to_flip = torch.rand((batch_size,))
    to_flip = (to_flip > 0.5)

    if not torch.any(to_flip):
        return images_batch, bboxes_batch

    images_to_flip = images_batch[to_flip, :, :, :]
    bboxes_to_flip = bboxes_batch[to_flip, :, :]
    images_flipped = []
    bboxes_flipped = []
    for image_single, bboxes_single in zip(images_to_flip, bboxes_to_flip):
        image_single_flipped, bboxes_single_flipped = flip_image(image_single, bboxes_single)
        images_flipped.append(image_single_flipped.unsqueeze(0))
        bboxes_flipped.append(bboxes_single_flipped.unsqueeze(0))

    images_flipped = torch.cat(images_flipped, dim=0)
    bboxes_flipped = torch.cat(bboxes_flipped, dim=0)

    images_batch[to_flip, :, :, :] = images_flipped
    bboxes_batch[to_flip, :, :] = bboxes_flipped

    return images_batch, bboxes_batch
