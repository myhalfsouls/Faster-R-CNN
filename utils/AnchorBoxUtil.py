import numpy as np
import torch
import torchvision
from tqdm import tqdm


def calculate_offsets(anchor_coords, pred_coords):
    # calculate offset as detailed in the paper
    anc = torchvision.ops.box_convert(anchor_coords, in_fmt='xyxy', out_fmt='cxcywh')
    pred = torchvision.ops.box_convert(pred_coords, in_fmt='xyxy', out_fmt='cxcywh')

    tx = (pred[:, 0] - anc[:, 0]) / anc[:, 2]
    ty = (pred[:, 1] - anc[:, 1]) / anc[:, 3]
    tw = torch.log(pred[:, 2] / anc[:, 2])
    th = torch.log(pred[:, 3] / anc[:, 3])

    return torch.stack([tx, ty, tw, th]).transpose(0, 1)


def evaluate_anchor_bboxes(all_anchor_bboxes, all_truth_bboxes, all_truth_labels, pos_thresh, neg_thresh, device='cpu'):
    batch_size = len(all_anchor_bboxes)
    num_anchor_bboxes_per = np.prod(list(all_anchor_bboxes.shape)[1:-1])
    num_anchor_bboxes = np.prod(list(all_anchor_bboxes.shape[1:4]))
    max_objects = all_truth_labels.shape[1]

    # get the complete IoU set
    iou_set = torch.zeros((batch_size, num_anchor_bboxes, max_objects)).to(device)
    for idx, (anchor_bboxes, truth_bboxes) in enumerate(zip(all_anchor_bboxes, all_truth_bboxes)):
        iou_set[idx, :] = torchvision.ops.box_iou(anchor_bboxes.reshape(-1, 4), truth_bboxes)

    # get the max per label
    iou_max_per_label, _ = iou_set.max(dim=1, keepdim=True)

    # "positive" consists of any anchor box that is (at least) one of:
    # 1. the max IoU and a ground truth box
    # 2. above our threshold
    pos_mask = torch.logical_and(iou_set == iou_max_per_label, iou_max_per_label > 0)
    pos_mask = torch.logical_or(pos_mask, iou_set > pos_thresh)

    # get indices where we meet the criteria
    pos_inds_batch = torch.where(pos_mask)[0]
    pos_inds_flat = torch.where(pos_mask.reshape(-1, max_objects))[0]

    # get the IoU and corresponding truth box
    iou_max_per_bbox, iou_max_per_bbox_inds = iou_set.max(dim=-1)
    iou_max_per_bbox_flat = iou_max_per_bbox.flatten(start_dim=0, end_dim=1)

    # parse out the positive scores
    pos_scores = iou_max_per_bbox_flat[pos_inds_flat]

    # map the predicted labels
    labels_expanded = all_truth_labels.unsqueeze(dim=1).repeat(1, num_anchor_bboxes, 1)
    labels_flat = torch.gather(labels_expanded, -1, iou_max_per_bbox_inds.unsqueeze(-1)).squeeze(-1).flatten(start_dim=0, end_dim=1)
    pos_labels = labels_flat[pos_inds_flat]

    # map the predicted bboxes
    bboxes_expanded = all_truth_bboxes.unsqueeze(dim=1).repeat(1, num_anchor_bboxes, 1, 1)
    bboxes_flat = torch.gather(bboxes_expanded, -2, iou_max_per_bbox_inds.reshape(batch_size, num_anchor_bboxes, 1, 1).repeat(1, 1, 1, 4)).flatten(start_dim=0, end_dim=2)
    pos_bboxes = bboxes_flat[pos_inds_flat]

    # calculate offsets against predicted bboxes
    pos_offsets = calculate_offsets(all_anchor_bboxes.reshape(-1, 4)[pos_inds_flat], pos_bboxes)

    # determine the indices where we fail the negative threshold criteria
    neg_mask = iou_max_per_bbox_flat < neg_thresh
    neg_inds_flat = torch.where(neg_mask)[0]
    neg_inds_flat = neg_inds_flat[torch.randint(0, len(neg_inds_flat), (len(pos_inds_flat),))]

    # get positive and negative anchor bboxes so we have easy access
    pos_points = all_anchor_bboxes.reshape(-1, 4)[pos_inds_flat]
    neg_points = all_anchor_bboxes.reshape(-1, 4)[neg_inds_flat]

    return pos_inds_flat, neg_inds_flat, pos_scores, pos_offsets, pos_labels, pos_bboxes, pos_points, neg_points, pos_inds_batch


def evaluate_anchor_bboxes_old(all_anchor_bboxes, all_truth_bboxes, all_truth_labels, pos_thresh, neg_thresh):
    batch_size = len(all_anchor_bboxes)
    num_anchor_bboxes_per = np.prod(list(all_anchor_bboxes.shape)[1:-1])
    max_objects = all_truth_labels.shape[1]

    # evaluate IoUs
    pos_coord_inds, neg_coord_inds, pos_scores, pos_classes, pos_offsets = [], [], [], [], []
    for idx, (anchor_bboxes, truth_bboxes, truth_labels) in enumerate(tqdm(zip(all_anchor_bboxes, all_truth_bboxes, all_truth_labels), total=all_anchor_bboxes.shape[0], desc='Evaluating Anchor Boxes')):
        # calculate the IoUs
        anchor_bboxes_flat = anchor_bboxes.reshape(-1, 4)
        iou_set = torchvision.ops.box_iou(anchor_bboxes_flat, truth_bboxes)

        # get the max per category
        iou_max_per_label, _ = iou_set.max(dim=0, keepdim=True)
        iou_max_per_bbox, _ = iou_set.max(dim=1, keepdim=True)

        # "positive" consists of any anchor box that is (at least) one of:
        # 1. the max IoU and a ground truth box
        # 2. above our threshold
        pos_mask = torch.logical_and(iou_set == iou_max_per_label, iou_max_per_label > 0)
        pos_mask = torch.logical_or(pos_mask, iou_set > pos_thresh)
        pos_inds_flat = torch.where(pos_mask)[0]
        pos_inds = torch.unravel_index(pos_inds_flat, all_anchor_bboxes.shape[1:4])
        pos_inds = torch.tensor([pos_ind.tolist() for pos_ind in pos_inds]).transpose(0, 1)
        pos_coord_inds.append(pos_inds)

        # "negative" consists of any anchor box whose max is below the threshold
        neg_mask = iou_max_per_bbox < neg_thresh
        neg_inds_flat = torch.where(neg_mask)[0]
        neg_inds_flat = neg_inds_flat[torch.randint(0, len(neg_inds_flat), (len(pos_inds_flat),))]
        neg_inds = torch.unravel_index(neg_inds_flat, all_anchor_bboxes.shape[1:4])
        neg_inds = torch.tensor([neg_ind.tolist() for neg_ind in neg_inds]).transpose(0, 1)
        neg_coord_inds.append(neg_inds)

        # get the IoU scores
        pos_scores.append(iou_max_per_bbox[pos_inds_flat])

        # get the classifications
        pos_indices = iou_set.argmax(dim=1)[pos_inds_flat]
        pos_classes_i = truth_labels[pos_indices]
        pos_classes.append(pos_classes_i)

        # calculate the offsets
        pos_offsets.append(calculate_offsets(anchor_bboxes_flat[pos_inds_flat], truth_bboxes[pos_indices]))

    return pos_coord_inds, neg_coord_inds, pos_scores, pos_classes, pos_offsets


def generate_anchors(h, w, device='cpu', resolution=10):

    # determine anchor points
    anc_pts_x = (torch.arange(0, w) + 0.5).to(device)
    anc_pts_y = (torch.arange(0, h) + 0.5).to(device)
    # w_steps = int(w / resolution)
    # h_steps = int(h / resolution)
    # anc_pts_x = torch.linspace(0, w, w_steps)[1:-1]
    # anc_pts_y = torch.linspace(0, h, h_steps)[1:-1]

    return anc_pts_x, anc_pts_y


def generate_anchor_boxes(h, w, scales, ratios, device):

    # determine anchor points
    anc_pts_x, anc_pts_y = generate_anchors(h, w, device)

    # initialize tensor for anchors
    n_boxes_per = len(scales) * len(ratios)
    anchor_boxes = torch.zeros(len(anc_pts_x), len(anc_pts_y), n_boxes_per, 4).to(device)

    for x_i, x in enumerate(anc_pts_x):
        for y_i, y in enumerate(anc_pts_y):
            boxes = torch.zeros((n_boxes_per, 4))

            ctr = 0
            for scale in scales:
                for ratio in ratios:
                    h_box = scale
                    w_box = scale * ratio
                    boxes[ctr, :] = torch.tensor([x - w_box / 2, y - h_box / 2, x + w_box / 2, y + h_box / 2])
                    ctr += 1
            anchor_boxes[x_i, y_i, :] = torchvision.ops.clip_boxes_to_image(boxes, (h, w))

    return anchor_boxes


def scale_bboxes(bboxes, h_scale, w_scale):
    bboxes_scaled = bboxes.clone()
    pad_mask = bboxes_scaled == -1
    if len(bboxes.shape) == 2:
        bboxes_scaled[:, [0, 2]] *= w_scale
        bboxes_scaled[:, [1, 3]] *= h_scale
    elif len(bboxes.shape) == 3:
        bboxes_scaled[:, :, [0, 2]] *= w_scale
        bboxes_scaled[:, :, [1, 3]] *= h_scale
    else:
        bboxes_scaled[:, :, :, [0, 2]] *= w_scale
        bboxes_scaled[:, :, :, [1, 3]] *= h_scale
    bboxes_scaled.masked_fill_(pad_mask, -1)
    return bboxes_scaled


def gen_k_center_anchors(sizes, aspect_ratios):
    """
    Generates k 0-centered anchors, where k = len(scales) x len(aspect_ratios)
    :param sizes: tuple of scales, defined as sqrt(H x W)
    :param aspect_ratios: tuple of aspect_ratios, defined as W:H
    :return: tensor of anchors, k x 4, in format x1, y1, x2, y2
    """

    # sizes = torch.as_tensor(sizes, dtype=torch.float32)
    # aspect_ratios = torch.as_tensor(aspect_ratios)
    # h_ratios = torch.sqrt(aspect_ratios)
    # w_ratios = 1.0 / h_ratios
    #
    # h_ratios = h_ratios.unsqueeze(1)
    # w_ratios = w_ratios.unsqueeze(1)
    # sizes = sizes.unsqueeze(0)
    #
    # w = torch.matmul(w_ratios, sizes).reshape(-1)
    # h = torch.matmul(h_ratios, sizes).reshape(-1)
    #
    # base_anchors = torch.stack([-w, -h, w, h], dim=1) / 2

    sizes = np.array(sizes).reshape((1, -1))
    aspect_ratios = np.array(aspect_ratios).reshape((-1, 1))
    h_ratios = np.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    w = w_ratios @ sizes
    h = h_ratios @ sizes
    w = w.flatten()
    h = h.flatten()

    base_anchors = np.stack([-w, -h, w, h], axis=1) / 2
    return base_anchors


def get_anchors(img, features, k_center_anchors):
    """
    Generates the anchors on a given input image
    :param img: input image
    :param features: feature map corresponding to input image. This is assumed to be the output of the final conv layer
    :return: anchors corresponding to each input image, tensor (k * feature_W * feature_H) x 4
    """

    k = k_center_anchors.shape[0]
    # k_center_anchors_shifted = np.zeros((k, 4))
    # k_center_anchors_shifted[:, 0:2] = -0.5 * k_center_anchors
    # k_center_anchors_shifted[:, 2:4] = 0.5 * k_center_anchors

    image_h, image_w = img.shape[-2:]
    feature_h, feature_w = features.shape[-2:]
    image_interval_w = int(image_w / feature_w)
    image_interval_h = int(image_h / feature_h)

    image_centers_w = np.arange(feature_w) * image_interval_w
    image_centers_h = np.arange(feature_h) * image_interval_h
    image_centers_temp = np.array(np.meshgrid(image_centers_w, image_centers_h))
    image_centers = image_centers_temp.transpose([1, 2, 0])

    image_centers1 = np.tile(image_centers, reps=2)
    image_centers2 = np.tile(image_centers1, reps=k)
    anchors = (image_centers2 + k_center_anchors.flatten())
    anchors1 = anchors.reshape(-1, 4)
    anchor_within_image_mask = np.all((anchors1[:, 0:2] >= [0, 0]) & (anchors1[:, 2:4] <= [image_w, image_h]), axis=1)
    anchors2 = torch.tensor(anchors1, dtype=torch.float32)
    anchor_within_image_mask = torch.tensor(anchor_within_image_mask)


    # image_h, image_w = img.shape[-2:]
    # feature_h, feature_w = features.shape[-2:]
    # image_interval_w = int(image_w / feature_w)
    # image_interval_h = int(image_h / feature_h)
    #
    # image_centers_w = torch.arange(0, feature_w) * image_interval_w
    # image_centers_h = torch.arange(0, feature_h) * image_interval_h
    #
    # image_centers_w, image_centers_h = torch.meshgrid(image_centers_w, image_centers_h)
    #
    # image_centers_w = image_centers_w.reshape(-1)
    # image_centers_h = image_centers_h.reshape(-1)
    #
    # image_centers = torch.stack([image_centers_w, image_centers_h, image_centers_w, image_centers_h], dim=1)
    #
    # k = k_center_anchors.shape[0]
    # num_centers = image_centers.shape[0]
    #
    # image_centers = image_centers.repeat_interleave(k, dim=0)
    # k_center_anchors = k_center_anchors.repeat(num_centers, 1)
    #
    # anchors = image_centers + k_center_anchors

    return anchors2, anchor_within_image_mask


def get_anchors_batch(img_all, sizes, aspect_ratios, features_all, device='cpu'):
    """
    Generates a tensor of anchors corresponding to the batch of input images
    :param img_all: a tensor containing a batch of input images: (num images) x 3 x H x W
    :param sizes: tuple of scales, defined as sqrt(H x W)
    :param aspect_ratios: tuple of aspect_ratios, defined as W:H
    :param features_all: features from the feature extractor
    :return: A tensor of anchors along with a tensor of extracted features for use in other networks
    """

    batch_size = img_all.shape[0]
    k_center_anchors = gen_k_center_anchors(sizes, aspect_ratios)
    anchors, anchor_within_image_mask = get_anchors(img_all[0, :, :, :], features_all[0, :, :, :], k_center_anchors)
    anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    return anchors, anchor_within_image_mask


def evaluate_anchor_bboxes_alt(all_anchor_bboxes, all_truth_bboxes, all_truth_labels, pos_thresh=0.7, neg_thresh=0.3, output_batch=256, pos_fraction=0.5):

    num_pos = int(output_batch * pos_fraction)
    num_neg = output_batch - num_pos

    # evaluate IoUs
    pos_coord_inds, neg_coord_inds, pos_scores, pos_classes, pos_offsets, pos_anchors = [], [], [], [], [], []

    for idx, (anchor_bboxes, truth_bboxes, truth_labels) in enumerate(zip(all_anchor_bboxes, all_truth_bboxes, all_truth_labels)):
        # calculate the IoUs
        # anchor_bboxes_flat = anchor_bboxes.reshape(-1, 4)
        truth_bboxes = truth_bboxes[torch.all((truth_bboxes != -1), dim=1), :]
        iou_set = torchvision.ops.box_iou(anchor_bboxes, truth_bboxes) # iou matrix, (num anchors) x (gt bboxes)

        # get the max per category
        iou_max_per_label, _ = iou_set.max(dim=0, keepdim=True)
        iou_max_per_bbox, pos_indices_per_bbox = iou_set.max(dim=1, keepdim=True)


        # "positive" consists of any anchor box that is (at least) one of:
        # 1. the max IoU and a ground truth box
        # 2. above our threshold
        pos_mask = torch.logical_and(iou_set == iou_max_per_label, iou_max_per_label > 0)
        pos_mask = torch.logical_or(pos_mask, iou_set > pos_thresh)
        pos_inds_flat = torch.where(pos_mask)[0]

        # pos_coord_inds.append(pos_inds_flat)

        if len(pos_inds_flat) > num_pos:
            # from pos_inds_flat, randomly sample num_pos samples without replacement
            rand_idx_pos = torch.randperm(len(pos_inds_flat))
            pos_inds_flat = pos_inds_flat[rand_idx_pos][0 : num_pos]
            pos_coord_inds.append(pos_inds_flat)
        else:
            # take all positive samples
            pos_coord_inds.append(pos_inds_flat)

        # "negative" consists of any anchor box whose max is below the threshold
        neg_mask = iou_max_per_bbox < neg_thresh
        neg_inds_flat = torch.where(neg_mask)[0]

        # rand_idx_neg = torch.randperm(len(neg_inds_flat))
        # neg_inds_flat = neg_inds_flat[rand_idx_neg][0: len(pos_inds_flat)]
        # neg_coord_inds.append(neg_inds_flat)

        if len(pos_inds_flat) > num_pos:
            # from neg_inds_flat, randomly sample num_neg samples without replacement
            rand_idx_neg = torch.randperm(len(neg_inds_flat))
            neg_inds_flat = neg_inds_flat[rand_idx_neg][0 : num_neg]
            neg_coord_inds.append(neg_inds_flat)
        else:
            # pad with negative samples
            rand_idx_neg = torch.randperm(len(neg_inds_flat))
            neg_inds_flat = neg_inds_flat[rand_idx_neg][0 : (num_neg + num_pos - len(pos_inds_flat))]
            neg_coord_inds.append(neg_inds_flat)

        # get the IoU scores
        pos_scores.append(iou_max_per_bbox[pos_inds_flat])

        # get the classifications
        pos_indices = pos_indices_per_bbox[pos_inds_flat]
        pos_classes_i = truth_labels[pos_indices].squeeze()
        pos_classes.append(pos_classes_i)

        # calculate the offsets
        pos_offsets.append(boxes_to_delta(anchor_bboxes[pos_inds_flat], truth_bboxes[pos_indices].squeeze(1)))

        pos_anchors.append(anchor_bboxes[pos_inds_flat])

    return pos_coord_inds, neg_coord_inds, pos_scores, pos_classes, pos_offsets, pos_anchors

def delta_to_boxes(rpn_delta, anchors):
    """
    Applies learned deltas from RPN network to the anchors to generate predicted boxes
    :param rpn_delta: learned deltas from RPN network (t_x, t_y, t_w, t_h), dimensions: num_anchors x 4
    :param anchors: anchors on input image (x1, y1, x2, y2)
    :return: predicted boxes
    """
    anchors_xywh = torchvision.ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    x_a = anchors_xywh[:, 0::4]
    y_a = anchors_xywh[:, 1::4]
    w_a = anchors_xywh[:, 2::4]
    h_a = anchors_xywh[:, 3::4]

    t_x = rpn_delta[:, 0::4]
    t_y = rpn_delta[:, 1::4]
    t_w = rpn_delta[:, 2::4]
    t_h = rpn_delta[:, 3::4]

    t_w = torch.clamp(t_w, max=4)
    t_h = torch.clamp(t_h, max=4)

    pred_box_x = torch.multiply(t_x, w_a) + x_a
    pred_box_y = torch.multiply(t_y, h_a) + y_a
    pred_box_w = torch.multiply(w_a, torch.exp(t_w))
    pred_box_h = torch.multiply(h_a, torch.exp(t_h))

    pred_box = torch.cat((pred_box_x, pred_box_y, pred_box_w, pred_box_h), dim=1)
    pred_box = torchvision.ops.box_convert(pred_box, in_fmt='cxcywh', out_fmt='xyxy')

    return pred_box


def boxes_to_delta(anchor_coords, pred_coords):
    # calculate offset as detailed in the paper
    anc = torchvision.ops.box_convert(anchor_coords, in_fmt='xyxy', out_fmt='cxcywh')
    pred = torchvision.ops.box_convert(pred_coords, in_fmt='xyxy', out_fmt='cxcywh')
    tx = (pred[:, 0] - anc[:, 0]) / anc[:, 2]
    ty = (pred[:, 1] - anc[:, 1]) / anc[:, 3]
    tw = torch.log(pred[:, 2] / anc[:, 2])
    th = torch.log(pred[:, 3] / anc[:, 3])

    return torch.stack([tx, ty, tw, th]).transpose(0, 1)


def assign_class(proposals, bboxes, labels, bg_thresh=0.5):
    assigned_labels = []
    truth_deltas = []
    for idx, (proposal, bbox, label) in enumerate(zip(proposals, bboxes, labels)):
        bbox = bbox[torch.all((bbox != -1), dim=1), :]
        iou_set = torchvision.ops.box_iou(proposal, bbox) # iou matrix, (num proposal boxes) x (gt bboxes)
        iou_max_per_bbox, best_indices_per_bbox = iou_set.max(dim=1, keepdim=True)

        assigned_label = label[best_indices_per_bbox]
        assigned_box = bbox[best_indices_per_bbox].squeeze()

        bg_mask = iou_max_per_bbox < bg_thresh
        assigned_label[bg_mask] = 0

        assigned_labels.append(assigned_label.squeeze().detach())
        truth_deltas.append(boxes_to_delta(proposal, assigned_box).detach())

    return proposals, assigned_labels, truth_deltas


def generate_size_mask(proposal, min_w=8, min_h=8):
    proposal_cwh = torchvision.ops.box_convert(proposal, in_fmt='xyxy', out_fmt='cxcywh')
    return torch.logical_and(proposal_cwh[:, 2] > min_w, proposal_cwh[:, 3] > min_h)
