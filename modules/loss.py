import torch
import torch.nn.functional as F
import modules.mod_utls as m_utls


Tensor = torch.tensor 


def nll_loss(pred, target, pos_w: float=1.0):
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor)

    return loss_value, m_utls.to_np(loss_value)


def nll_loss_raw(pred: Tensor, target: Tensor, pos_w,
                reduction: str='mean'):
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor,
                           reduction=reduction)
    
    return loss_value
  
    
def l2_regularization(model):
    l2_reg = torch.tensor(0., requires_grad=True)
    for key, value in model.named_parameters():
        if len(value.shape) > 1 and 'weight' in key:
            l2_reg = l2_reg + torch.sum(value ** 2) * 0.5
    return l2_reg    


