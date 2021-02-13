
import config
import torch
from torch import nn
from torchvision.models import mobilenet_v2
import torch.nn.functional as F

# predict: (M, 4)
# ground_truth: (K, 4)
# returns: (K, M)
def computeIOU(predict, ground_truth):

    ground_truth = ground_truth.unsqueeze(1)
    predict = predict.unsqueeze(0)

    gt_area = ground_truth[:, :, 2] * ground_truth[:, :, 3]
    pd_area = predict[:, :, 2] * predict[:, :, 3]

    left = torch.max(ground_truth[:, :, 0] - ground_truth[:, :, 2] / 2, predict[:, :, 0] - predict[:, :, 2] / 2)
    right = torch.min(ground_truth[:, :, 0] + ground_truth[:, :, 2] / 2, predict[:, :, 0] + predict[:, :, 2] / 2)
    bottom = torch.max(ground_truth[:, :, 1] - ground_truth[:, :, 3] / 2, predict[:, :, 1] - predict[:, :, 3] / 2)
    top = torch.min(ground_truth[:, :, 1] + ground_truth[:, :, 3] / 2, predict[:, :, 1] + predict[:, :, 3] / 2)

    overlap_exists = (right > left) * (top > bottom)

    intersection = overlap_exists * (top - bottom) * (right - left)

    iou = intersection / (gt_area + pd_area - intersection)

    return iou

class MobileNetV2_SSD(nn.Module):
    def __init__(self):
        super(MobileNetV2_SSD, self).__init__()
        self.n_class = len(config.CLASSES)
        self.n_box = config.N_BOX
        self.neg_pos_ratio = config.NEGATIVE_POSTIVE_RATIO
        self.alpha = config.ALPHA
        self.iou_threshold = config.MIN_IOU_2POSTIVE
        self.class_prior = torch.FloatTensor(config.CLASS_PRIOR)
        self.box_prior = torch.FloatTensor(config.BOX_PRIOR)
        self.device = torch.device('cpu')

        self.input_size = (None, 3, 256, 256)
        self.featureExtracter = mobilenet_v2(pretrained=config.MOBILENET_PRETRAINED).features
        # self.auxilary1 = nn.Sequential(nn.Conv2d(1280, 64, 1, 1), nn.ReLU(inplace=True))
        self.auxilary2 = nn.Sequential(nn.Conv2d(1280, 640, 3, 2), nn.ReLU(inplace=True))
        self.auxilary3 = nn.Sequential(nn.Conv2d(640, 320, 3, 2), nn.ReLU(inplace=True))

        self.classifier0 = nn.Conv2d(1280, self.n_box * (self.n_class + 1), 3, 1, padding=1)
        self.detector0 = nn.Conv2d(1280, self.n_box * 4, 3, 1, padding=1)

        self.classifier1 = nn.Conv2d(640, self.n_box * (self.n_class + 1), 3, 1, padding=1)
        self.detector1 = nn.Conv2d(640, self.n_box * 4, 3, 1, padding=1)

        self.classifier2 = nn.Conv2d(320, self.n_box * (self.n_class + 1), 3, 1, padding=1)
        self.detector2 = nn.Conv2d(320, self.n_box * 4, 3, 1, padding=1)

    def to(self, device):
        new_self = super(MobileNetV2_SSD, self).to(device)
        new_self.class_prior = new_self.class_prior.to(device)
        new_self.box_prior = new_self.box_prior.to(device)
        new_self.device = device
        return new_self

    def forward(self, x):
        features = []
        x = self.featureExtracter(x)
        # x = self.auxilary1(x)
        features.append(x)
        x = self.auxilary2(x)
        features.append(x)
        x = self.auxilary3(x)
        features.append(x)

        cl_predict0 = self.classifier0(features[0])
        cl_predict0 = cl_predict0.permute(0, 2, 3, 1)
        cl_predict0 = cl_predict0.view(-1, 8 * 8 * self.n_box, self.n_class + 1)
        bb_predict0 = self.detector0(features[0])
        bb_predict0 = bb_predict0.permute(0, 2, 3, 1)
        bb_predict0 = bb_predict0.view(-1, 8 * 8 * self.n_box, 4)

        cl_predict1 = self.classifier1(features[1])
        cl_predict1 = cl_predict1.permute(0, 2, 3, 1)
        cl_predict1 = cl_predict1.view(-1, 3 * 3 * self.n_box, self.n_class + 1)
        bb_predict1 = self.detector1(features[1])
        bb_predict1 = bb_predict1.permute(0, 2, 3, 1)
        bb_predict1 = bb_predict1.view(-1, 3 * 3 * self.n_box, 4)

        cl_predict2 = self.classifier2(features[2])
        cl_predict2 = cl_predict2.permute(0, 2, 3, 1)
        cl_predict2 = cl_predict2.view(-1, 1 * 1 * self.n_box, self.n_class + 1)
        bb_predict2 = self.detector2(features[2])
        bb_predict2 = bb_predict2.permute(0, 2, 3, 1)
        bb_predict2 = bb_predict2.view(-1, 1 * 1 * self.n_box, 4)

        self.cl_predict = torch.cat([cl_predict0, cl_predict1, cl_predict2], dim=1) * self.class_prior.view(1, 1, self.n_class + 1) * (self.n_class + 1) # (N, M, C+1)
        self.bb_predict = torch.cat([bb_predict0, bb_predict1, bb_predict2], dim=1) + self.box_prior.unsqueeze(0) # (N, M, 4)

        return self.cl_predict, self.bb_predict

    # labels: (N, k)
    # boxes: (N, k, 4)
    def backward(self, labels, boxes):

        n_batch = self.cl_predict.size(0)
        n_proposal = self.cl_predict.size(1)
        label_per_proposal_all = torch.zeros(n_batch, n_proposal, dtype=torch.long, device=self.device)
        box_per_proposal_all = torch.zeros(n_batch, n_proposal, 4, dtype=torch.float, device=self.device)
        for i in range(n_batch):
            label = labels[i] # (k)
            box = boxes[i] # (k, 4)
            non_backgrounds = label != 0
            label = label[non_backgrounds]
            box = box[non_backgrounds]
            n_object = box.size(0)
            iou = computeIOU(self.bb_predict[i], box) # (k, M)
            iou_per_proposal, objectid_per_proposal = torch.max(iou, dim=0) # (M)

            # force at least 1 proposal per object to be postive
            _, max_proposal_per_objectid = torch.max(iou, dim=1) # (k)
            objectid_per_proposal[max_proposal_per_objectid] = torch.LongTensor(range(n_object)).to(self.device)
            iou_per_proposal[max_proposal_per_objectid] = 1.

            label_per_proposal = label[objectid_per_proposal] # (M)
            label_per_proposal[iou_per_proposal < self.iou_threshold] = 0
            box_per_proposal = box[objectid_per_proposal] # (M, 4)
            label_per_proposal_all[i] = label_per_proposal
            box_per_proposal_all[i] = box_per_proposal

        label_per_proposal_all = label_per_proposal_all.view(-1)
        positive_idx = label_per_proposal_all != 0 # (N * M)
        pos_bb_loss = F.l1_loss(self.bb_predict.view(-1, 4)[positive_idx], box_per_proposal_all.view(-1, 4)[positive_idx]) # (1)
        cl_loss = F.cross_entropy(self.cl_predict.view(-1, self.n_class + 1), label_per_proposal_all, reduction='none') #(N * M)
        cl_loss_pos = cl_loss[positive_idx]

        n_positive = positive_idx.sum() # (1)
        if n_proposal >= (n_positive * (self.neg_pos_ratio + 1)):
            n_hard_negative = n_positive * self.neg_pos_ratio # (1)
            n_hard_negative = n_hard_negative.type(torch.LongTensor)
        else:
            n_hard_negative = n_proposal - n_positive # (1)

        cl_loss_neg = cl_loss.clone()
        cl_loss_neg[positive_idx] = 0
        cl_loss_neg = cl_loss_neg.sort(descending=True)[0][:n_hard_negative]

        cl_loss_total = (cl_loss_pos.sum() + cl_loss_neg.sum()) / n_positive.float()

        total_loss = cl_loss_total + self.alpha * pos_bb_loss

        total_loss.backward()

        return (cl_loss_pos.sum() / n_positive.float()).item(), (cl_loss_neg.sum() / n_positive.float()).item(), pos_bb_loss.item()


    def detect(self, min_score=config.MIN_SCORE_2DETECTION,
                     max_iou=config.MAX_IOU,
                     top_k=config.TOP_K):

        device = self.cl_predict.device

        # min_score = config.MIN_SCORE_2DETECTION
        # max_iou = config.MAX_IOU
        # top_k = config.TOP_K

        n_batch = self.cl_predict.size(0)

        class_scores_all = F.softmax(self.cl_predict, dim=-1)

        all_image_boxes = []
        all_image_labels = []
        all_image_scores = []

        for i in range(n_batch):

            image_boxes = []
            image_labels = []
            image_scores = []

            for c in range(1, self.n_class + 1):

                detected_idx = class_scores_all[i, :, c] > min_score #(M+)
                class_scores = class_scores_all[i, detected_idx, c] #(M+)
                detected_loc = self.bb_predict[i, detected_idx] #(M+, 4))

                class_scores, sorted_ind = class_scores.sort(dim=0, descending=True) #(M+)
                detected_loc = detected_loc[sorted_ind] # (M+, 4)

                iou_score = computeIOU(detected_loc, detected_loc) # (M+, M+)
                mask = torch.ones(detected_loc.size(0), dtype=torch.bool, device=self.device)

                for b in range(detected_loc.size(0)):
                    if mask[b] is False:
                        continue

                    mask = mask * (iou_score[b] < max_iou)
                    mask[b] = True

                if mask.sum().item() > 0:
                    image_boxes.append(detected_loc[mask])
                    image_labels.append(c * torch.ones(mask.sum().item(), dtype=torch.long, device=self.device))
                    image_scores.append(class_scores[mask])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0.5, 0.5, 1., 1.]], device=self.device))
                image_labels.append(torch.LongTensor([0], device=self.device))
                image_scores.append(torch.FloatTensor([0.], device=self.device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            n_object = image_scores.size(0)

            if n_object > top_k:
                image_scores, sorted_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sorted_ind][:top_k]
                image_labels = image_labels[sorted_ind][:top_k]

            all_image_boxes.append(image_boxes)
            all_image_labels.append(image_labels)
            all_image_scores.append(image_scores)

        return all_image_boxes, all_image_labels, all_image_scores
