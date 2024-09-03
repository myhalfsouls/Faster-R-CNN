from networks.FRCRPN import FRCRPN
from networks.FCN import MaskHead
from networks.FRCClassifier import FRCClassifier

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

class MaskRCNN(nn.Module):
    def __init__(self, img_size, roi_size, n_labels, top_n,
                 pos_thresh=0.68, neg_thresh=0.30, nms_thresh=0.7, hidden_dim=512, dropout=0.1, backbone='resnet50', device='cpu'):
        super().__init__()

        self.hyper_params = {
            'img_size': img_size,
            'roi_size': roi_size,
            'n_labels': n_labels,
            'pos_thresh': pos_thresh,
            'neg_thresh': neg_thresh,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'backbone': backbone
        }

        self.device = device

        if backbone == 'resnet50':
            # resnet backbone
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            req_layers = list(model.children())[:8]
            self.backbone = nn.Sequential(*req_layers)
            for param in self.backbone.named_parameters():
                param[1].requires_grad = True
            self.backbone_size = (2048, 15, 20)
            self.feature_to_image_scale = 0.03125
        else:
            raise NotImplementedError

        # initialize the RPN and classifier
        self.rpn = FRCRPN(img_size, pos_thresh, neg_thresh, nms_thresh, top_n, self.backbone_size, hidden_dim, dropout, device=device).to(device)
        self.classifier = FRCClassifier(roi_size, self.backbone_size, n_labels, hidden_dim, dropout, device=device).to(device)
        self.mask_head = MaskHead(2048, num_classes=n_labels)
        self.mask_loss_fn = nn.BCELoss(reduction='none')

    def forward(self, images, truth_labels, truth_bboxes, truth_masks):
        features = self.backbone(images)
        
        # evaluate region proposal network
        rpn_loss, proposals, assigned_labels, _ = self.rpn(features, images, truth_labels, truth_bboxes)
        print('rpn done')

        # proposals_by_batch = []
        # for idx in range(images.shape[0]):
        #     batch_proposals = proposals[torch.where(pos_inds_batch == idx)[0]].detach().clone()
        #     proposals_by_batch.append(batch_proposals)

        true_label_count = truth_labels.ne(-1).sum(dim=1)
        # print(f"true label count: {true_label_count}")

        # print(f"len(proposals_by_batch): {len(proposals_by_batch)}")
        # print(f"proposals_by_batch[0].shape: {proposals_by_batch[0].shape}")
        # print(f"proposals_by_batch[1].shape: {proposals_by_batch[1].shape}")

        # perform ROI align for Mask R-CNN
        rois = torchvision.ops.roi_align(input=features,
                                         boxes=proposals,
                                         output_size=self.hyper_params["roi_size"],
                                         spatial_scale=self.feature_to_image_scale)

        print('roi done')

        # run classifier
        class_scores = self.classifier(rois)

        # calculate cross entropy loss
        class_loss = nn.functional.cross_entropy(class_scores, labels)

        print('class done')

        # print(f"truth_bboxes.shape: {truth_bboxes.shape}")
        # print(f"truth_labels.shape: {truth_labels.shape}")
        # print(f"truth_labels: {truth_labels}")
        # print(f"rois.shape: {rois.shape}")
        
        masks = self.mask_head(rois)
        
        # TODO: Parse masks to only calculate loss on mask for ground truth class
        # TODO: Remove padding from ground truth masks
        
        # print(f"masks.shape: {masks.shape}")
        # print(f"truth_masks.shape: {truth_masks.shape}")
        
        # mask_loss = self.mask_loss_fn(masks, truth_masks)
        mask_loss = 0

        return rpn_loss + class_loss + mask_loss

    def evaluate(self, images, confidence_thresh=0.5, nms_thresh=0.7):
        features = self.backbone(images)
        
        proposals_by_batch, scores = self.rpn.evaluate(features, images)
        
        # perform ROI align for Mask R-CNN
        rois = torchvision.ops.roi_align(input=features,
                                         boxes=proposals_by_batch,
                                         output_size=self.hyper_params["roi_size"])
        
        class_scores = self.classifier(rois)

        # evaluate using softmax
        p = nn.functional.softmax(class_scores, dim=-1)
        preds = p.argmax(dim=-1)

        labels = []
        i = 0
        for proposals in proposals_by_batch:
            n = len(proposals)
            labels.append(preds[i: i + n])
            i += n

        return proposals_by_batch, scores, labels
