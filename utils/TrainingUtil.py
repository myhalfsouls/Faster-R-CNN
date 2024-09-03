from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
from networks.FasterRCNN import FasterRCNN
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import torch
import time
from tqdm import tqdm
from utils import EvalUtil, ImageUtil
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_model(model, optimizer, data_train, data_val, num_epochs, batch_size, id2str, device='cpu', verbose=True, save=True, horizontal_flip=False, metrics=False):
    quiet = not verbose

    can_debug = isinstance(model, FasterRCNN)

    # training loop
    model.train()
    loss_tracker = []
    val_loss_tracker = []
    mAP_avg_tracker = []
    mAP_per_tracker = []
    best_epoch, best_mAP, best_model = None, None, None
    train_losses_tracker = {'rpn_class': [], 'rpn_box': [], 'cls_class': [], 'cls_box': []}
    val_losses_tracker = {'rpn_class': [], 'rpn_box': [], 'cls_class': [], 'cls_box': []}
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=True)
    
    for i in tqdm(range(1, num_epochs + 1), disable=verbose, desc='Training Model', file=sys.stdout):

        if verbose:
            print("-" * 60)
            print("Running Epoch: %02d / %02d" % (i, num_epochs))

        # evaluate per batch
        loss = 0
        train_losses = {'rpn_class': 0, 'rpn_box': 0, 'cls_class': 0, 'cls_box': 0}
        for data in tqdm(dataloader, disable=quiet, desc='Training Model', file=sys.stdout):
            # Send data to CUDA device
            if horizontal_flip:
                images_batch, labels_batch, bboxes_batch = data
                images_batch, bboxes_batch = ImageUtil.random_flip_batch(images_batch, bboxes_batch)
                data = (images_batch, labels_batch, bboxes_batch)
            data_device = []
            for item in data:
                # Segmentation masks are not stored as Tensors because they are all different shapes
                if isinstance(item, torch.Tensor):
                    item = item.to(device)
                data_device.append(item)
            data = data_device
            # forward
            if can_debug:
                data.append(True)
                epoch_loss, epoch_losses = model(*data)
                for loss_type in train_losses.keys():
                    train_losses[loss_type] += epoch_losses[loss_type]
            else:
                epoch_loss = model(*data)

            # backward
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()

            loss += epoch_loss.item()

        # compute validation loss
        model.eval()
        val_loss = 0

        batch_truth_boxes = []
        batch_truth_labels = []
        batch_eval_boxes = []
        batch_eval_labels = []
        val_losses = {'rpn_class': 0, 'rpn_box': 0, 'cls_class': 0, 'cls_box': 0}
        if metrics:
            mAP = MeanAveragePrecision()
        with torch.no_grad():
            if metrics:
                preds = []
                targets = []
            for data in tqdm(dataloader_val, disable=quiet, desc='Running Validation', file=sys.stdout):
                data_device = []
                for item in data:
                    # Segmentation masks are not stored as Tensors because they are all different shapes
                    if isinstance(item, torch.Tensor):
                        item = item.to(device)
                    data_device.append(item)
                data = data_device
                if can_debug:
                    data.append(True)
                    val_loss_epoch, epoch_losses = model(*data)
                    for loss_type in val_losses.keys():
                        val_losses[loss_type] += epoch_losses[loss_type]
                else:
                    val_loss_epoch = model(*data)

                val_loss += val_loss_epoch.item()

                # compute validation mAP
                if metrics:
                    batch_dict = model.evaluate(data[0], device=device, metrics=True)
                    batch_gt = []

                    for i in range(batch_size):
                        gt_labels = data[1][i, :]
                        gt_boxes = data[2][i, :, :]
                        gt_labels = gt_labels[gt_labels != -1]
                        gt_boxes = gt_boxes[torch.all((gt_boxes != -1), dim=1), :]
                        gt = {'boxes': gt_boxes, 'labels': gt_labels}
                        batch_gt.append(gt)

                    preds = preds + batch_dict
                    targets = targets + batch_gt
                else:
                    proposals, labels = model.evaluate(data[0], device=device, metrics=False)

                    batch_truth_boxes += [entry.tolist() for entry in data[2]]
                    batch_truth_labels += [entry.tolist() for entry in data[1]]
                    batch_eval_boxes += [entry.tolist() for entry in proposals]
                    batch_eval_labels += [entry.tolist() for entry in labels]
        if metrics:
            mAP.update(preds, targets)
            result = mAP.compute()
            val_mAP = result['map'].item()
            ap_dict = {}
        else:
            val_mAP, ap_dict = EvalUtil.model_eval(id2str,
                                                   batch_truth_boxes,
                                                   batch_truth_labels,
                                                   batch_eval_boxes,
                                                   batch_eval_labels)
        val_loss = val_loss / data_val.n_samples

        model.train()

        # scheduler.step()

        # track loss
        loss /= data_train.n_samples
        loss_tracker.append(loss)
        val_loss_tracker.append(val_loss)
        mAP_avg_tracker.append(val_mAP * 100)
        mAP_per_tracker.append(ap_dict)
        if verbose:
            print(ap_dict)
            print("  Training Loss: %.2f, Validation Loss %.2f, Validation mAP %.4f" % (loss, val_loss, val_mAP*100))

        # save the best model
        if (best_mAP is None) or (val_mAP > best_mAP):
            best_epoch = i
            best_mAP = val_mAP
            best_model = deepcopy(model.state_dict())

        # loss breakdown
        if can_debug:
            for loss_type in train_losses_tracker.keys():
                train_losses_tracker[loss_type].append(train_losses[loss_type] / data_train.n_samples)
                val_losses_tracker[loss_type].append(val_losses[loss_type] / data_val.n_samples)

    if save:
        # make a save directory
        save_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base_dir = os.path.join("results/models", save_timestamp)
        Path(base_dir).mkdir(parents=True, exist_ok=True)

        # save the model
        model_filename = os.path.join(base_dir, "model.pt")
        torch.save(model.state_dict(), model_filename)
        best_model_filename = os.path.join(base_dir, "model_epoch_{}.pt".format(best_epoch))
        torch.save(best_model, best_model_filename)

        # save the properties
        save_properties(model, optimizer, base_dir)

        # save the loss curve
        save_loss_curve(loss_tracker, val_loss_tracker, base_dir, save_timestamp)

        # save mAP information
        save_mAP(mAP_avg_tracker, mAP_per_tracker, base_dir, save_timestamp)

        # save loss breakdown
        if can_debug:
            save_losses_curve(train_losses_tracker, val_losses_tracker, base_dir, save_timestamp)

        print("Model, properties, and results saved to: {}".format(base_dir))
    return loss_tracker


def save_properties(model, optimizer, base_dir):

    props_filename = os.path.join(base_dir, "training.properties")
    props_file = open(props_filename, "w")

    # write the model properties
    for param in model.hyper_params.keys():
        props_file.write("{}={}\n".format("model.{}".format(param), model.hyper_params[param]))

    # write the optimizer properties
    for param in ['lr', 'momentum']:
        props_file.write("{}={}\n".format("optim.{}".format(param), optimizer.param_groups[0][param]))

    props_file.close()


def save_loss_curve(loss_tracker, val_loss_tracker, base_dir, save_timestamp):
    loss_curve_filename = os.path.join(base_dir, "loss_{}.png".format(save_timestamp))
    epochs = np.arange(1, len(loss_tracker) + 1, dtype=int).tolist()
    plt.cla()
    plt.plot(epochs, loss_tracker, label='Training')
    plt.plot(epochs, val_loss_tracker, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(loss_curve_filename)
    plt.cla()

    loss_filename = os.path.join(base_dir, "loss_{}_values.csv".format(save_timestamp))
    dataset = np.array([np.arange(1, len(loss_tracker) + 1).tolist(), loss_tracker, val_loss_tracker]).transpose().tolist()
    dataframe = pd.DataFrame(dataset, columns=['epoch', 'train_loss', 'val_loss'])
    dataframe.to_csv(loss_filename, index=False)


def save_mAP(mAP_avg_tracker, mAP_per_tracker, base_dir, save_timestamp):
    epochs = np.arange(1, len(mAP_avg_tracker) + 1, dtype=int).tolist()

    mAP_curve_filename = os.path.join(base_dir, "mAP_{}.png".format(save_timestamp))
    plt.cla()
    plt.plot(epochs, mAP_avg_tracker)
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP")
    plt.grid(True)
    plt.savefig(mAP_curve_filename)
    plt.cla()

    mAP_filename = os.path.join(base_dir, "mAP_{}_values.csv".format(save_timestamp))
    mAP_list = [epochs, mAP_avg_tracker]
    mAP_cols = ['epoch', 'avg']
    for key in mAP_per_tracker[0].keys():
        mAP_list.append([mAP_per_tracker[i][key] for i in range(len(mAP_per_tracker))])
        mAP_cols.append(key)
    dataset = np.array(mAP_list).transpose().tolist()
    dataframe = pd.DataFrame(dataset, columns=mAP_cols)
    dataframe.to_csv(mAP_filename, index=False)


def save_losses_curve(train_losses_tracker, val_losses_tracker, base_dir, save_timestamp):
    loss_curve_filename = os.path.join(base_dir, "losses_{}.png".format(save_timestamp))
    epochs = np.arange(1, len(train_losses_tracker['rpn_class']) + 1, dtype=int).tolist()
    plt.cla()
    data, cols = [], ['epoch']
    for loss_type in train_losses_tracker.keys():
        plt.plot(epochs, train_losses_tracker[loss_type], label="Train {}".format(loss_type))
        data.append(train_losses_tracker[loss_type])
        cols.append("train_{}_loss".format(loss_type))
    for loss_type in val_losses_tracker.keys():
        plt.plot(epochs, val_losses_tracker[loss_type], label="Val {}".format(loss_type))
        data.append(val_losses_tracker[loss_type])
        cols.append("val_{}_loss".format(loss_type))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(loss_curve_filename)
    plt.cla()

    loss_filename = os.path.join(base_dir, "losses_{}_values.csv".format(save_timestamp))
    dataset = np.array([np.arange(1, len(train_losses_tracker['rpn_class']) + 1).tolist()] + data).transpose().tolist()
    dataframe = pd.DataFrame(dataset, columns=cols)
    dataframe.to_csv(loss_filename, index=False)
