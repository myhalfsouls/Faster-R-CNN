import torch
import torch.nn as nn
import torchvision
from utils import AnchorBoxUtil


# the Faster R-CNN (FRC) Region Proposal Network


class FRCRPN(nn.Module):
    def __init__(self, img_size, pos_thresh, neg_thresh, nms_thresh, top_n, backbone_size, hidden_dim=512, dropout=0.1, anc_scales=None, anc_ratios=None, loss_scale=1, device='cpu'):
        super().__init__()

        self.device = device

        # store dimensions
        self.h_inp, self.w_inp = img_size

        # positive/negative thresholds for IoU
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.nms_thresh = nms_thresh

        self.top_n = top_n
        self.loss_scale = loss_scale

        # scales and ratios
        if anc_scales is None:
            anc_scales = [64, 128, 256, 512]
        if anc_ratios is None:
            anc_ratios = [0.5, 1.0, 2.0]
        self.scales = anc_scales
        self.ratios = anc_ratios

        # proposal network
        self.c_out, self.h_out, self.w_out = backbone_size # feature channels, feature h, feature w
        self.num_anchors_per = len(self.scales) * len(self.ratios)
        self.proposal = nn.Sequential(
            nn.Conv2d(self.c_out, hidden_dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.regression = nn.Conv2d(hidden_dim, self.num_anchors_per * 4, kernel_size=1)
        self.confidence = nn.Conv2d(hidden_dim, self.num_anchors_per, kernel_size=1)
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.l1_loss = nn.SmoothL1Loss(reduction='sum')

        # image scaling
        self.h_scale = self.h_inp / self.h_out
        self.w_scale = self.w_inp / self.w_out

    def forward(self, features, images, labels, bboxes, debug=False):
        # bboxes = the ground truth boxes, batch_size x num_boxes x 4, tensors are padded with -1 when there are not enough bboxes

        batch_size = images.shape[0]

        # generate anchor boxes
        anchors_all, anchor_within_image_mask = AnchorBoxUtil.get_anchors_batch(images, self.scales, self.ratios, features, device=self.device)
        # anchors_all = torchvision.ops.clip_boxes_to_image(anchors_all, images.shape[-2:])
        anchors_all = anchors_all[:, anchor_within_image_mask, :]
        anchors_single = anchors_all[0, :, :]

        # evaluate for positive and negative anchors
        pos_coord_inds, neg_coord_inds, pos_scores, pos_classes, pos_offsets, pos_anchors = AnchorBoxUtil.evaluate_anchor_bboxes_alt(anchors_all, bboxes, labels, pos_thresh=0.7, neg_thresh=0.3, output_batch=256, pos_fraction=0.5)

        # evaluate with proposal network
        proposal = self.proposal(features)
        regression = self.regression(proposal) # batch_size x 4*k x feature h x feature w
        regression = regression.permute(0, 2, 3, 1)
        regression = regression.reshape(batch_size, -1, 4)
        regression = regression[:, anchor_within_image_mask, :]

        confidence = self.confidence(proposal) # batch_size x 1*k x feature h x feature w
        confidence = confidence.permute(0, 2, 3, 1)
        confidence = confidence.reshape(batch_size, -1)
        confidence = confidence[:, anchor_within_image_mask]

        total_loss = 0
        top_proposals = []
        # top_confidences = []
        losses = {'rpn_class': 0, 'rpn_box': 0}

        for i in range(batch_size):

            pos_inds_flat = pos_coord_inds[i]
            neg_inds_flat = neg_coord_inds[i]
            pos_offset = pos_offsets[i]
            regression_i = regression[i, :, :]
            confidence_i = confidence[i, :]

            # parse out confidence
            pos_confidence = confidence[i, pos_inds_flat]
            neg_confidence = confidence[i, neg_inds_flat]

            # parse out regression
            pos_regression = regression[i, pos_inds_flat, :]

            # calculate loss
            target = torch.cat((torch.ones_like(pos_confidence), torch.zeros_like(neg_confidence)))
            scores = torch.cat((pos_confidence, neg_confidence))
            class_loss = self.ce_loss(scores, target) / target.shape[0]
            bbox_loss = self.l1_loss(pos_regression, pos_offset) * self.loss_scale / (pos_offset.shape[0] + 1e-7)
            total_loss = total_loss + class_loss + bbox_loss
            losses['rpn_class'] += class_loss.item()
            losses['rpn_box'] += bbox_loss.item()

            proposal = AnchorBoxUtil.delta_to_boxes(regression_i, anchors_single)

            prenms_topn = 12000
            if confidence_i.shape[0] < prenms_topn:
                prenms_topn = confidence_i.shape[0]

            top_confidence_sorted, top_indices_sorted = torch.topk(confidence_i, prenms_topn, dim=0)
            proposal = proposal[top_indices_sorted]

            proposal = torchvision.ops.clip_boxes_to_image(proposal, images.shape[-2:])

            size_mask = AnchorBoxUtil.generate_size_mask(proposal)
            proposal = proposal[size_mask]
            top_confidence_sorted = top_confidence_sorted[size_mask]

            nms_mask = torchvision.ops.nms(proposal, top_confidence_sorted, self.nms_thresh)
            proposal = proposal[nms_mask]
            proposal = proposal[0:self.top_n]

            top_proposals.append(proposal.detach())

        filtered_proposals, assigned_labels, truth_deltas = AnchorBoxUtil.assign_class(top_proposals, bboxes, labels, bg_thresh=0.4)
        if debug:
            return total_loss, filtered_proposals, assigned_labels, truth_deltas, losses
        return total_loss, filtered_proposals, assigned_labels, truth_deltas

    def evaluate(self, features, images, top_n):

        batch_size = images.shape[0]

        # generate anchor boxes
        anchors_all, _ = AnchorBoxUtil.get_anchors_batch(images, self.scales, self.ratios, features, device=self.device)
        anchors_single = anchors_all[0, :, :]

        # evaluate with proposal network
        proposal = self.proposal(features)
        regression = self.regression(proposal)  # batch_size x 4*k x feature h x feature w
        regression = regression.permute(0, 2, 3, 1)
        regression = regression.reshape(batch_size, -1, 4)

        confidence = self.confidence(proposal)  # batch_size x 1*k x feature h x feature w
        confidence = confidence.permute(0, 2, 3, 1)
        confidence = confidence.reshape(batch_size, -1)

        top_proposals = []

        for i in range(batch_size):

            regression_i = regression[i, :, :]
            confidence_i = confidence[i, :]

            proposal = AnchorBoxUtil.delta_to_boxes(regression_i, anchors_single)

            prenms_topn = 12000
            if confidence_i.shape[0] < prenms_topn:
                prenms_topn = confidence_i.shape[0]

            top_confidence_sorted, top_indices_sorted = torch.topk(confidence_i, prenms_topn, dim=0)
            proposal = proposal[top_indices_sorted]

            proposal = torchvision.ops.clip_boxes_to_image(proposal, images.shape[-2:])

            size_mask = AnchorBoxUtil.generate_size_mask(proposal)
            proposal = proposal[size_mask]
            top_confidence_sorted = top_confidence_sorted[size_mask]

            nms_mask = torchvision.ops.nms(proposal, top_confidence_sorted, 0.7)

            proposal = proposal[nms_mask]
            proposal = proposal[0:top_n]

            top_proposals.append(proposal)

        return top_proposals

    # def evaluate(self, features, images, confidence_thresh=0.5, nms_thresh=0.7):
    #     batch_size = images.shape[0]
    #
    #     # generate anchor boxes
    #     anchors_all, _ = AnchorBoxUtil.get_anchors_batch(images, self.scales, self.ratios, features, device=self.device)
    #     anchors_all = torchvision.ops.clip_boxes_to_image(anchors_all, images.shape[-2:])
    #     anchors_single = anchors_all[0, :, :]
    #
    #     # evaluate with proposal network
    #     proposal = self.proposal(features)
    #     regression = self.regression(proposal)  # batch_size x 4*k x feature h x feature w
    #     regression = regression.permute(0, 2, 3, 1)
    #     regression = regression.reshape(batch_size, -1, 4)
    #
    #     confidence = self.confidence(proposal)  # batch_size x 1*k x feature h x feature w
    #     confidence = confidence.permute(0, 2, 3, 1)
    #     confidence = confidence.reshape(batch_size, -1, 1).squeeze()
    #
    #     proposals, scores = [], []
    #     for confidence_i, regression_i, batch_anchor_bboxes in zip(confidence, regression, anchors_all):
    #         confidence_score = torch.sigmoid(confidence_i)
    #         proposals_i = AnchorBoxUtil.delta_to_boxes(regression_i, anchors_single)
    #         confidence_mask = confidence_score > confidence_thresh
    #         proposals_i = proposals_i[confidence_mask]
    #         confidence_score = confidence_score[confidence_mask]
    #         nms_mask = torchvision.ops.nms(proposals_i, confidence_score, nms_thresh)
    #         scores.append(confidence_score[nms_mask])
    #         proposals_i = proposals_i[nms_mask]
    #
    #         proposals.append(proposals_i)
    #
    #     return proposals, scores

    @staticmethod
    def generate_proposals(pos_points, pos_regression):
        pos_points_cxcywh = torchvision.ops.box_convert(pos_points, in_fmt='xyxy', out_fmt='cxcywh')
        proposals = torch.zeros_like(pos_points)
        proposals[:, 0] = pos_points_cxcywh[:, 0] + pos_regression[:, 0] * pos_points_cxcywh[:, 2]
        proposals[:, 1] = pos_points_cxcywh[:, 1] + pos_regression[:, 1] * pos_points_cxcywh[:, 3]
        proposals[:, 2] = pos_points_cxcywh[:, 2] + torch.exp(pos_points_cxcywh[:, 2])
        proposals[:, 3] = pos_points_cxcywh[:, 3] + torch.exp(pos_points_cxcywh[:, 3])
        return torchvision.ops.box_convert(pos_points, in_fmt='cxcywh', out_fmt='xyxy')




class FRCRPN_old(nn.Module):
    def __init__(self, img_size, pos_thresh, neg_thresh, backbone_size, hidden_dim=512, dropout=0.1, device='cpu'):
        super().__init__()

        self.device = device

        # store dimensions
        self.h_inp, self.w_inp = img_size

        # positive/negative thresholds for IoU
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        # scales and ratios
        self.scales = [2, 4, 6]
        self.ratios = [0.5, 1.0, 1.5]

        # proposal network
        self.c_out, self.h_out, self.w_out = backbone_size
        self.num_anchors_per = len(self.scales) * len(self.ratios)
        self.proposal = nn.Sequential(
            nn.Conv2d(self.c_out, hidden_dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.regression = nn.Conv2d(hidden_dim, self.num_anchors_per * 4, kernel_size=1)
        self.confidence = nn.Conv2d(hidden_dim, self.num_anchors_per, kernel_size=1)
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.l1_loss = nn.SmoothL1Loss(reduction='sum')

        # image scaling
        self.h_scale = self.h_inp / self.h_out
        self.w_scale = self.w_inp / self.w_out

    def forward(self, features, images, labels, bboxes):
        batch_size = images.shape[0]

        # generate anchor boxes
        anchor_bboxes = AnchorBoxUtil.generate_anchor_boxes(self.h_out, self.w_out, self.scales, self.ratios, device=self.device)
        all_anchor_bboxes = anchor_bboxes.repeat(batch_size, 1, 1, 1, 1)

        # evaluate for positive and negative anchors
        bboxes_scaled = AnchorBoxUtil.scale_bboxes(bboxes, 1 / self.h_scale, 1 / self.w_scale)
        pos_inds_flat, neg_inds_flat, pos_scores, pos_offsets, pos_labels, pos_bboxes, pos_points, neg_points, pos_inds_batch = AnchorBoxUtil.evaluate_anchor_bboxes(all_anchor_bboxes, bboxes_scaled, labels, self.pos_thresh, self.neg_thresh, device=self.device)

        # evaluate with proposal network
        proposal = self.proposal(features)
        regression = self.regression(proposal)
        confidence = self.confidence(proposal)

        # parse out confidence
        pos_confidence = confidence.flatten()[pos_inds_flat]
        neg_confidence = confidence.flatten()[neg_inds_flat]

        # parse out regression and convert to proposals
        pos_regression = regression.reshape(-1, 4)[pos_inds_flat]
        proposals = self.generate_proposals(pos_points, pos_regression)

        # calculate loss
        target = torch.cat((torch.ones_like(pos_confidence), torch.zeros_like(neg_confidence)))
        scores = torch.cat((pos_confidence, neg_confidence))
        class_loss = self.ce_loss(scores, target) / batch_size
        bbox_loss = self.l1_loss(pos_offsets, pos_regression) / batch_size
        total_loss = class_loss + bbox_loss

        return total_loss, proposals, pos_labels, pos_inds_batch

    def evaluate(self, features, images, confidence_thresh=0.5, nms_thresh=0.7):
        batch_size = images.shape[0]

        # generate anchor boxes
        anchor_bboxes = AnchorBoxUtil.generate_anchor_boxes(self.h_out, self.w_out, self.scales, self.ratios, device=self.device)
        all_anchor_bboxes = anchor_bboxes.repeat(batch_size, 1, 1, 1, 1)
        all_anchor_bboxes_batched = all_anchor_bboxes.reshape(batch_size, -1, 4)

        # evaluate with proposal network
        proposal = self.proposal(features)
        regression = self.regression(proposal).reshape(batch_size, -1, 4)
        confidence = self.confidence(proposal).reshape(batch_size, -1)

        proposals, scores = [], []
        for confidence_i, regression_i, batch_anchor_bboxes in zip(confidence, regression, all_anchor_bboxes_batched):
            confidence_score = torch.sigmoid(confidence_i)
            proposals_i = self.generate_proposals(batch_anchor_bboxes, regression_i)
            confidence_mask = confidence_score > confidence_thresh
            proposals_i = proposals_i[confidence_mask]
            confidence_score = confidence_score[confidence_mask]
            nms_mask = torchvision.ops.nms(proposals_i, confidence_score, nms_thresh)
            scores.append(confidence_score[nms_mask])
            proposals_i = proposals_i[nms_mask]

            # scale up to the image dimensions and clip
            proposals_i = AnchorBoxUtil.scale_bboxes(proposals_i, self.h_scale, self.w_scale)
            proposals_i = torchvision.ops.clip_boxes_to_image(proposals_i, (self.h_inp, self.w_inp))
            proposals.append(proposals_i)

        return proposals, scores

    @staticmethod
    def generate_proposals(pos_points, pos_regression):
        pos_points_cxcywh = torchvision.ops.box_convert(pos_points, in_fmt='xyxy', out_fmt='cxcywh')
        proposals = torch.zeros_like(pos_points)
        proposals[:, 0] = pos_points_cxcywh[:, 0] + pos_regression[:, 0] * pos_points_cxcywh[:, 2]
        proposals[:, 1] = pos_points_cxcywh[:, 1] + pos_regression[:, 1] * pos_points_cxcywh[:, 3]
        proposals[:, 2] = pos_points_cxcywh[:, 2] + torch.exp(pos_points_cxcywh[:, 2])
        proposals[:, 3] = pos_points_cxcywh[:, 3] + torch.exp(pos_points_cxcywh[:, 3])
        return torchvision.ops.box_convert(pos_points, in_fmt='cxcywh', out_fmt='xyxy')
