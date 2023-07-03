# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion
from fairseq.criterions.hubert_criterion import HubertCriterionConfig,HubertCriterion

@register_criterion("fasthubert", dataclass=HubertCriterionConfig)
class FasthubertCriterion(HubertCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        log_keys=None,
    ):
        super().__init__(task,pred_masked_weight,pred_nomask_weight,loss_weights,log_keys)
  
    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"]) 
        loss = 0.0   
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True) 
        targ_m_list = net_output["targ_m_list"] 
        logp_m_list= torch.cat(logp_m_list) 
        targ_m_list= torch.cat(targ_m_list) 

        loss_m = F.cross_entropy(logp_m_list, targ_m_list, reduction=reduction)  
        loss_m_list.append(loss_m)
        logging_output[f"loss_m_0"] = loss_m.detach().item()
        
        if self.pred_masked_weight > 0: 
            loss += self.pred_masked_weight * sum(loss_m_list) 
            sample_size += len(targ_m_list)

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = net_output["targ_u_list"]
        logp_u_list= torch.cat(logp_u_list )  
        targ_u_list= torch.cat(targ_u_list) 

        loss_u = F.cross_entropy(logp_u_list, targ_u_list, reduction=reduction)
        loss_u_list.append(loss_u)
        logging_output[f"loss_u_0"] = loss_u.detach().item()
        
        if self.pred_nomask_weight > 0:  
            loss += self.pred_nomask_weight * sum(loss_u_list)   
            sample_size += len(targ_u_list)

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys: 
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        with torch.no_grad():    
            count_m= len(logp_m_list)
            max_index = torch.argmax(logp_m_list,dim=1)
            corr_m= ( max_index==targ_m_list ).sum().item()
                
            logging_output[f"correct_m_0"] = corr_m
            logging_output[f"count_m_0"] = count_m
            
            count_u= len(logp_u_list)
            max_index = torch.argmax(logp_u_list,dim=1)
            corr_u = ( max_index==targ_u_list ).sum().item()   
              
            logging_output[f"correct_u_0"] = corr_u
            logging_output[f"count_u_0"] = count_u

        return loss, sample_size, logging_output
