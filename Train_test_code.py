from networks.FasterRCNN import FasterRCNN
from utils import AnchorBoxUtil, DataManager, ImageUtil, TrainingUtil
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print('Current device is: {}'.format(device))

# data parameters
dataset_name = "voc-2007"
num_train = None
num_val = None
num_test = None
h_img_std = 480 # standard image height to resize to
w_img_std = 640 # standard image width to resize to
data_train, data_val, str2id, id2str = DataManager.load_supplemented_data(dataset_name, num_train, num_val, (h_img_std, w_img_std))

# # example for using ImageUtil
# example_idx = 0
# # id_cats = {cat_ids[key]: key for key in cat_ids.keys()}                            # reverse the category ids
# example_image = data_train.images[example_idx].long()                              # look up the image and convert to long
# example_labels = [id2str[key] for key in data_train.labels[example_idx].tolist()] # get the actual label string(s)
# example_bboxes = data_train.bboxes[example_idx]                                    # get this sample's bounding box(es) (format is: [x_min, y_min, x_max, y_max])
# ImageUtil.build_image(example_image, example_bboxes, example_labels, 'g', show=True)


# parameters
img_size = (h_img_std, w_img_std)
roi_size = (7, 7)
n_labels = len(str2id) - 1
pos_thresh = 0.7
neg_thresh = 0.3
nms_thresh = 0.7
top_n = 128
hidden_dim_rpn = 512
hidden_dim_class = 512
dropout = 0.2
# backbone_size = (2048, 15, 20)
backbone = 'resnet50'
loss_scale = 1

model = FasterRCNN(img_size, roi_size, n_labels, top_n, pos_thresh, neg_thresh, nms_thresh, hidden_dim_rpn, hidden_dim_class, dropout, backbone, loss_scale=loss_scale, device=device)

# parameters
learning_rate = 1e-3
momentum = 0.9
weight_decay = 5e-4
num_epochs = 100
batch_size = 10

# initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# initialize scheduler
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [11], gamma=0.1)

# run training
loss_results = TrainingUtil.train_model(model, optimizer, data_train, data_val, num_epochs, batch_size, id2str, device=device, save=True, horizontal_flip=True, metrics=True)

# model.load_state_dict(torch.load('G:\\Dropbox (GaTech)\\CS7643_DL\\Mask_R-CNN\\results\\models\\balanced losses_final layer_freezes\\model.pt'))

#
#
#
#
# import sys
# import torch
# from tqdm import tqdm
# from utils import EvalUtil
#
# quiet = not True
#
# model.eval()
# batch_truth_boxes = []
# batch_truth_labels = []
# batch_eval_boxes = []
# batch_eval_labels = []
# dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=True)
# mAP = MeanAveragePrecision()
# with torch.no_grad():
#     preds = []
#     targets = []
#     for data in tqdm(dataloader, disable=quiet, desc='Running Validation', file=sys.stdout):
#         data_device = []
#         for item in data:
#             # Segmentation masks are not stored as Tensors because they are all different shapes
#             if isinstance(item, torch.Tensor):
#                 item = item.to(device)
#             data_device.append(item)
#         data = data_device
#         # if can_debug:
#         #     data.append(True)
#         #     val_loss_epoch, epoch_losses = model(*data)
#         #     for loss_type in val_losses.keys():
#         #         val_losses[loss_type] += epoch_losses[loss_type]
#         # else:
#         #     val_loss_epoch = model(*data)
#         #
#         # val_loss += val_loss_epoch.item()
#
#         # compute validation mAP
#         batch_dict = model.evaluate(data[0], device=device, metrics=True)
#         batch_gt = []
#
#         for i in range(batch_size):
#             gt_labels = data[1][i, :]
#             gt_boxes = data[2][i, :, :]
#             gt_labels = gt_labels[gt_labels != -1]
#             gt_boxes = gt_boxes[torch.all((gt_boxes != -1), dim=1), :]
#             gt = {'boxes': gt_boxes, 'labels': gt_labels}
#             batch_gt.append(gt)
#
#         preds = preds + batch_dict
#         targets = targets + batch_gt
#
# mAP.update(preds, targets)
# result = mAP.compute()
# print(result)