from networks.FRCRPN import FRCRPN
from networks.FRCClassifier import FRCClassifier_fasteronly

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

from utils import AnchorBoxUtil


class FasterRCNN(nn.Module):
    def __init__(self, img_size, roi_size, n_labels, top_n,
                 pos_thresh=0.68, neg_thresh=0.30, nms_thresh=0.7, hidden_dim_rpn=2048, hidden_dim_class=512, dropout=0.1, backbone='resnet50',
                 anc_scales=None, anc_ratios=None, loss_scale=1, device='cpu'):
        super().__init__()

        self.hyper_params = {
            'img_size': img_size,
            'roi_size': roi_size,
            'n_labels': n_labels,
            'pos_thresh': pos_thresh,
            'neg_thresh': neg_thresh,
            'nms_thresh': nms_thresh,
            'hidden_dim_rpn': hidden_dim_rpn,
            'hidden_dim_class': hidden_dim_class,
            'dropout': dropout,
            'backbone': backbone,
            'anc_scales': anc_scales,
            'anc_ratios': anc_ratios,
            'loss_scale': loss_scale
        }

        self.device = device

        if backbone == 'resnet50':
            # resnet backbone
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            req_layers = list(model.children())[:7]
            self.backbone = nn.Sequential(*req_layers).eval().to(device)
            for param in self.backbone.named_parameters():
                param[1].requires_grad = True
            self.backbone_classifier = list(model.children())[7]
            for param in self.backbone_classifier.named_parameters():
                param[1].requires_grad = True

            # Freeze initial layers
            for param in self.backbone[0].named_parameters():
                param[1].requires_grad = False
            for param in self.backbone[1].named_parameters():
                param[1].requires_grad = False
            for param in self.backbone[4].named_parameters():
                param[1].requires_grad = False

            # Freeze batchnorm layers
            for child in self.backbone.modules():
                if type(child) == nn.BatchNorm2d:
                    for param in child.named_parameters():
                        param[1].requires_grad = False

            for child in self.backbone_classifier.modules():
                if type(child) == nn.BatchNorm2d:
                    for param in child.named_parameters():
                        param[1].requires_grad = False

            self.backbone_size = (1024, 30, 40)
            self.feature_to_image_scale = 0.0625
        else:
            raise NotImplementedError

        # initialize the RPN and classifier
        self.rpn = FRCRPN(img_size, pos_thresh, neg_thresh, nms_thresh, top_n, self.backbone_size, hidden_dim_rpn, dropout, anc_scales, anc_ratios, loss_scale, device=device).to(device)
        self.classifier = FRCClassifier_fasteronly(roi_size, self.backbone_size, n_labels, self.feature_to_image_scale, self.backbone_classifier, hidden_dim_class, dropout, loss_scale, device=device).to(device)

    def forward(self, images, truth_labels, truth_bboxes, debug=False):
        # with torch.no_grad():
        features = self.backbone(images)

        if debug:
            rpn_loss, proposals, assigned_labels, truth_deltas, rpn_losses = self.rpn(features, images, truth_labels, truth_bboxes, debug)
            class_loss, class_losses = self.classifier(features, proposals, assigned_labels, truth_deltas, debug)
            losses = {'rpn_class': rpn_losses['rpn_class'], 'rpn_box': rpn_losses['rpn_box'], 'cls_class': class_losses['cls_class'], 'cls_box': class_losses['cls_box']}
            return rpn_loss + class_loss, losses

        # evaluate region proposal network
        rpn_loss, proposals, assigned_labels, truth_deltas = self.rpn(features, images, truth_labels, truth_bboxes)

        # proposals_by_batch = []
        # for idx in range(images.shape[0]):
        #     batch_proposals = proposals[torch.where(pos_inds_batch == idx)[0]].detach().clone()
        #     proposals_by_batch.append(batch_proposals)

        # run classifier
        class_loss = self.classifier(features, proposals, assigned_labels, truth_deltas)

        return rpn_loss + class_loss

    def evaluate(self, images, top_n=300, confidence_thresh=0.05, nms_thresh_final=0.3, device='cpu', metrics=False):
        features = self.backbone(images)

        proposals_by_batch = self.rpn.evaluate(features, images, top_n)
        class_scores, box_deltas = self.classifier.evaluate(features, proposals_by_batch)

        batch_proposals = []
        batch_labels = []
        batch_scores = []
        for idx, proposals_i in enumerate(proposals_by_batch):
            class_proposals_dict = {}
            class_scores_dict = {}
            scores_i = class_scores[idx * top_n:(idx + 1) * top_n, :]
            deltas_i = box_deltas[idx * top_n:(idx + 1) * top_n, :]

            for class_idx in range(1, scores_i.shape[1]):  # skip background
                scores_i_class = scores_i[:, class_idx]
                deltas_i_class = deltas_i[:, class_idx * 4:(class_idx + 1) * 4]
                select_mask = torch.where(scores_i_class > confidence_thresh)[0]
                if select_mask.numel() == 0:
                    continue
                proposals_i_class_select = proposals_i[select_mask]
                deltas_i_class_select = deltas_i_class[select_mask]
                scores_i_class_select = scores_i_class[select_mask]
                proposals_i_class_select = AnchorBoxUtil.delta_to_boxes(deltas_i_class_select, proposals_i_class_select)
                proposals_i_class_select = torchvision.ops.clip_boxes_to_image(proposals_i_class_select,
                                                                               images.shape[-2:])

                nms_mask = torchvision.ops.nms(proposals_i_class_select, scores_i_class_select, nms_thresh_final)

                class_proposals_dict[class_idx] = proposals_i_class_select[nms_mask]
                class_scores_dict[class_idx] = scores_i_class_select[nms_mask]

            filtered_proposals = []
            filtered_labels = []
            filtered_scores = []

            for cls in class_proposals_dict.keys():
                prop = class_proposals_dict[cls]
                src = class_scores_dict[cls]
                filtered_labels = filtered_labels + [cls] * prop.shape[0]
                filtered_proposals.append(prop)
                filtered_scores.append(src)

            if len(filtered_proposals) == 0:
                filtered_proposals = torch.tensor(filtered_proposals)
                filtered_scores = torch.tensor(filtered_scores)
            else:
                filtered_proposals = torch.cat(filtered_proposals)
                filtered_scores = torch.cat(filtered_scores)
            filtered_labels = torch.tensor(filtered_labels)

            batch_proposals.append(filtered_proposals)
            batch_labels.append(filtered_labels.to(device))
            batch_scores.append(filtered_scores.to(device))

        if metrics:
            metric_dict_list = []
            for single_proposals, single_labels, single_scores in zip(batch_proposals, batch_labels, batch_scores):
                metric_dict = {'boxes': single_proposals, 'scores': single_scores, 'labels': single_labels}
                metric_dict_list.append(metric_dict)
            return metric_dict_list
        else:
            return batch_proposals, batch_labels

    # def evaluate(self, images, confidence_thresh=0.8, nms_thresh=0.4, device='cpu'):
    #     features = self.backbone(images)
    #
    #     proposals_by_batch, scores = self.rpn.evaluate(features, images, confidence_thresh, nms_thresh)
    #     class_scores, box_deltas = self.classifier.evaluate(features, proposals_by_batch)
    #
    #     # evaluate using softmax
    #     p = nn.functional.softmax(class_scores, dim=-1)
    #     preds = p.argmax(dim=-1)
    #
    #     labels = []
    #     box_deltas_list = []
    #     i = 0
    #     for proposals in proposals_by_batch:
    #         n = len(proposals)
    #         labels.append(preds[i: i + n])
    #         box_deltas_list.append(box_deltas[i: i + n, :])
    #         i += n
    #
    #     final_proposals, final_labels = [], []
    #     for (proposals, label, deltas, score) in zip(proposals_by_batch, labels, box_deltas_list, scores):
    #         fg_mask = (label != 0)
    #         proposals = proposals[fg_mask, :]
    #         deltas = deltas[fg_mask, :]
    #         label = label[fg_mask]
    #         score = score[fg_mask]
    #
    #         truth_deltas = torch.zeros((label.shape[0], 4)).to(device)
    #         for i in range(label.shape[0]):
    #             gt = label[i]
    #             truth_deltas[i, :] = deltas[i, (4 * gt):(4 * gt + 4)]
    #
    #         final_proposal = AnchorBoxUtil.delta_to_boxes(truth_deltas, proposals)
    #         final_proposal = torchvision.ops.clip_boxes_to_image(final_proposal, images.shape[-2:])
    #
    #         size_mask = AnchorBoxUtil.generate_size_mask(final_proposal, min_w=5, min_h=5)
    #         final_proposal = final_proposal[size_mask]
    #         label = label[size_mask]
    #         score = score[size_mask]
    #
    #         nms_mask = torchvision.ops.nms(final_proposal, score, nms_thresh)
    #
    #         final_proposals.append(final_proposal[nms_mask])
    #         final_labels.append(label[nms_mask])
    #
    #     return final_proposals, final_labels #proposals_by_batch, scores, labels
