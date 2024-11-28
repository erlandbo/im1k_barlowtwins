import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SACLRAll(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
        self.projector = nn.Sequential(*layers)

        ################################
        # SACLR
        self.register_buffer("s_inv", torch.zeros(1 if args.single_s else args.N, ) + 1.0 / args.s_init)
        self.register_buffer("N", torch.zeros(1, ) + args.N)
        self.register_buffer("rho", torch.zeros(1, ) + args.rho)
        self.register_buffer("alpha", torch.zeros(1, ) + args.alpha)
        self.temp = args.temp
        self.single_s = args.single_s
        self.distributed = args.distributed


    def forward(self, y1, y2, feats_idx):
        #print(feats_idx)

        feats_a = self.projector(self.backbone(y1))
        feats_b = self.projector(self.backbone(y2))
        
        ###################
        # SACLR
        LARGE_NUM = 1e9
        B = feats_a.shape[0]
        # if self.normalize
        feats_a = F.normalize(feats_a, dim=1, p=2)
        feats_b = F.normalize(feats_b, dim=1, p=2)

        if self.distributed:  
           feats_a_large = torch.cat(all_gather_layer.apply(feats_a), dim=0) # why concat_all_gather()
           feats_b_large =  torch.cat(all_gather_layer.apply(feats_b), dim=0)
           enlarged_B = feats_a_large.shape[0]

           labels_idx = (torch.arange(B, dtype=torch.long) + B  * torch.distributed.get_rank()).to(feats_a.device) 
           labels = F.one_hot(labels_idx, enlarged_B*2).to(feats_a.device) 
           masks  = F.one_hot(labels_idx, enlarged_B).to(feats_a.device) 
           B = enlarged_B
        else:
           feats_a_large = feats_a
           feats_b_large = feats_b
           labels = F.one_hot(torch.arange(B, dtype=torch.long), B * 2).to(feats_a.device) 
           masks  = F.one_hot(torch.arange(B, dtype=torch.long), B).to(feats_a.device)            
               
        #feats_a = feats[:B]
        #feats_b = feats[B:]
        #print(feats_a_large.shape)
        #print(feats_b_large.shape)
        #print(feats_idx.shape)

        #masks = F.one_hot(torch.arange(B, device=feats_a.device), B)

        logits_aa = -1.0 * torch.cdist(feats_a, feats_a_large, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = -1.0 * torch.cdist(feats_b, feats_b_large, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = -1.0 * torch.cdist(feats_a, feats_b_large, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_ba = -1.0 * torch.cdist(feats_b, feats_a_large, p=2).pow(2) / (2.0 * self.temp **2.0)

        logits_pos_a = torch.diag(logits_ab)
        logits_pos_b = torch.diag(logits_ba)

        logits_ab = logits_ab - masks * LARGE_NUM
        logits_ba = logits_ba - masks * LARGE_NUM

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        
        q_attr_a = torch.exp(logits_pos_a)  # .clamp(min=1e-21) # (B,)
        q_attr_b = torch.exp(logits_pos_b)  #.clamp(min=1e-21) # (B,)

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)

        q_rep_a = torch.exp(logits_a)  #.clamp(min=1e-21) # (B,2B)
        q_rep_b = torch.exp(logits_b)  #.clamp(min=1e-21) # (B,2B)

        if self.single_s:
            feats_idx = 0
        else:
            feats_idx = feats_idx

        with torch.no_grad():
            Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2) 

        repulsive_forces_a = torch.sum(q_rep_a / Z_hat.detach().view(-1,1), dim=1) 
        repulsive_forces_b = torch.sum(q_rep_b / Z_hat.detach().view(-1,1), dim=1) 

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        ##############################
        # Update s
        # self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)
        if self.single_s:
            feats_idx = 0
        else:
            if self.distributed:
                feats_idx = concat_all_gather(feats_idx)
            else:
                feats_idx = feats_idx

        B = q_attr_a.size(0)

        E_attr_a = q_attr_a.detach()  
        E_attr_b = q_attr_b.detach()  

        E_rep_a = torch.sum(q_rep_a.detach(), dim=1) / (2.0 * B - 2.0)  
        E_rep_b = torch.sum(q_rep_b.detach(), dim=1) / (2.0 * B - 2.0)  
        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B  

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a 
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        if self.distributed:
            s_inv_a_large = concat_all_gather(s_inv_a)
            s_inv_b_large = concat_all_gather(s_inv_b)
            self.s_inv[feats_idx] = (s_inv_a_large + s_inv_b_large) / 2.0
        else:
            self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0

        return loss


# Copied from https://github.com/Optimization-AI/SogCLR/blob/PyTorch/sogclr/builder.py

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class all_gather_layer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out