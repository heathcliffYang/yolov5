# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj, llandmark = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
    MSElandmarks = nn.MSELoss()

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            # print(pi.shape)
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss
            # landmark regression
            plandmarks = ps[:, 5+model.nc:].sigmoid() * 2. - 0.5
            glandmarks = tbox[i][:,4:]
            no_landmarks_f = ~(glandmarks == torch.zeros_like(glandmarks[0])).all(1)
            plandmarks = plandmarks[no_landmarks_f]
            glandmarks = glandmarks[no_landmarks_f]
            llandmark += MSElandmarks(plandmarks, glandmarks)
            print("llandmark:", llandmark)


            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:5+model.nc], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:5+model.nc], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    llandmark *= h['box'] * s * 0.1
    print(lbox.shape, llandmark.shape)
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls + llandmark
    return loss * bs, torch.cat((lbox, lobj, lcls, loss, llandmark)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    # print(na, nt)
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    # print(gain.shape, ai.shape, targets.shape)

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        # print(anchors.shape)
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        # print(targets[:, :, :6].shape, targets[:, :, -1][:,:,None].shape)
        t = torch.cat((targets[:, :, :6], targets[:, :, -1][:,:,None]), 2)  * gain
        # print(gain[2:4].repeat(68).shape, targets[:, :, 7:].shape)
        landmark_xy = targets[:, :, 6:16] * gain[2:4].repeat(5)
        print("162:", landmark_xy.shape)

        if nt:
            # Matches
            # print(anchors[:, None].shape)
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            # print("torch max:", torch.max(r, 1. / r).shape)
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            # print(j.shape, t.shape)
            t = t[j]  # filter
            landmark_xy = landmark_xy[j]
            print("174:", landmark_xy.shape)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            # print(j.shape, t.repeat((5, 1, 1)).shape)
            t = t.repeat((5, 1, 1))[j]
            landmark_xy = landmark_xy.repeat((5, 1, 1))[j]
            print("185:", landmark_xy.shape)
            # print(t.shape)
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            # print(torch.zeros_like(gxy)[None].shape, off[:, None].shape, (torch.zeros_like(gxy)[None] + off[:, None]).shape)
        else:
            t = targets[0]
            landmark_xy = landmark_xy[0]
            print("No?", t.shape, landmark_xy.shape)
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        # print("image ??", b, c)
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        # aaaaa = torch.tensor([0., 0.2, 0.7, 1., 1.2, 4.8])
        # print(aaaaa.long())
        gij = (gxy - offsets).long()
        # print("gxy:", gxy)
        # print("gij:", gij)
        # print("offset:", offsets)
        gi, gj = gij.T  # grid xy indices

        # print(gij.shape, gij.repeat(1,68).shape)
        glandmarks = landmark_xy - gij.repeat(1,5)
        print("208:", glandmarks.shape)
        # print(t[100000000000000])

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh, glandmarks), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch
