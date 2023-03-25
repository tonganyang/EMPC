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
    
    mask = [1 for _ in range(len(logits2))]
    
    for i in range(len(logits2)):
        
        logits = logits2[i].detach()
        pseudo_label = torch.softmax(logits, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mas = max_probs.ge(p_cutoff).float()
        mask = map(lambda x,y:x*y, mask, mas)
    
    return (supervised_loss(logits1, max_idx.detach(), use_hard_labels=True,
                    reduction='none') * mask).mean()

def uncertainty_loss(logit_S):
    # the Jensen-Shannon divergence between p(x) and mean
    S = nn.Softmax(dim=1)
    mean = sum([S(i) for i in logit_S]) / len(logit_S)
    kl_divergence = nn.KLDivLoss()
    JSloss = sum([kl_divergence(i.log(), mean) for i in logit_S]) / len(logit_S)
    
    return JSloss

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
