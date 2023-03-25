# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def supervised_loss(logits, targets, use_hard_labels=True, reduction='none'):
    
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def unsupervised_loss(logits1, logits2, p_cutoff):
    logits2 = logits2.detach()
    pseudo_label = torch.softmax(logits2, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    # 计算伪标签
    mask = max_probs.ge(p_cutoff).float()
    
    select = max_probs.ge(p_cutoff).long()
    # 计算无监督损失
    return (supervised_loss(logits1, max_idx.detach(), use_hard_labels=True,
                    reduction='none') * mask).mean(), select, max_idx.long()

def uncertainty_loss(logit_S):
    # the Jensen-Shannon divergence between p(x) and mean
    length = len(logit_S)
    S = nn.Softmax(dim=1)
    mean = sum([S(i) for i in logit_S]) / length
    kl_divergence = nn.KLDivLoss()
    JSloss = sum([kl_divergence(i.log(), mean) for i in logit_S]) / length
    
    return JSloss

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device)) # 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss