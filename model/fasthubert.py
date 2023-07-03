# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fairseq.data.dictionary import Dictionary

from fairseq.models import register_model
from fairseq.models.speech_to_text.modules.convolution import Conv1dSubsampler
from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform

from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from ..task.hubert_fbank_pretraining import HubertFbankPretrainingTask
from fairseq.models.hubert.hubert import HubertModel,HubertConfig

logger = logging.getLogger(__name__)


@dataclass
class FastHubertConfig(HubertConfig):  
    # Downsample layer config
    input_feat_per_channel: int = field(default=80, metadata={"help": "dimension of input features of each channel"})
    
    input_channels: int = field(default=1, metadata={"help": "number of input channels"})
    
    conv_channels: int = field(default=1024, metadata={"help": "number of intermediate channels"})
    
    fbank_encoder_dim: int = field(default=512, metadata={"help": "number of output channels"})
    
    conv_kernel_sizes: str = field(default="5,5", metadata={"help": "kernel size for each convolutional layer"})
    
    # ILS config
    ils: bool = field(default=True, metadata={"help": "use intermediate layer supervision mechanism or not"})
    
    predict_layers: str = field(default="[3,11]", metadata={"help": "intermediate layer set"})
    
    # SpecAugment config
    freq_mask_F: int = field(default=30, metadata={"help": "maximum value of the mask width in frequency domain"})
    
    freq_mask_N: int = field(default=2, metadata={"help": "number of masks in frequency domain"})
    
    time_mask_N: int = field(default=2, metadata={"help": "maximum value of the mask width in time domain"})
    
    time_mask_T: int = field(default=40, metadata={"help": "number of masks in time domain"})
    
    time_mask_p: float = field(default=1.0, metadata={"help": "mask probability in time domain"})
    
    time_wrap_W: int = field(default=0, metadata={"help": "warp boundary"})

        
@register_model("fasthubert", dataclass=FastHubertConfig)
class FastHubertModel(HubertModel):
    def __init__(
        self,
        cfg: FastHubertConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__(cfg, task_cfg, dictionaries)
        logger.info(f"FastHubertModelConfig: {cfg}") 

        self.embed = cfg.fbank_encoder_dim 
        self.feat2tar_ratio = 2** (len(cfg.conv_kernel_sizes.split(","))-1 )              
        self.post_extract_proj = (  
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )
        self.num_classes = len(dictionaries[0]) 
        
        # Downsample
        self.subsample = Conv1dSubsampler(
            cfg.input_feat_per_channel * cfg.input_channels,
            cfg.conv_channels, cfg.fbank_encoder_dim,
            [int(k) for k in cfg.conv_kernel_sizes.split(",")] 
        )
        
        # SpecAugment   
        specaug_config = {"freq_mask_F": cfg.freq_mask_F, "freq_mask_N": cfg.freq_mask_N, "time_mask_N": cfg.time_mask_N, "time_mask_T": cfg.time_mask_T, "time_mask_p": cfg.time_mask_p, "time_wrap_W": cfg.time_wrap_W}
        self.specaug_transform = SpecAugmentTransform.from_config_dict(specaug_config)

        # ILS
        self.ils=cfg.ils
        self.predict_layers = eval(cfg.predict_layers)   
        if self.ils:
            del self.final_proj
            self.final_proj = torch.nn.Sequential(
                *[nn.Linear(cfg.encoder_embed_dim, self.num_classes) 
                for _ in range(len(self.predict_layers))]
            )  
            
    @classmethod
    def build_model(cls, cfg: FastHubertConfig, task: HubertFbankPretrainingTask):
        """Build a new model instance."""

        model = FastHubertModel(cfg, task.cfg, task.dictionaries)
        return model


    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""

        self.fine_tuning=features_only  
        if self.fine_tuning==True and self.encoder.training == True:  
            for i in range(source.shape[0]):  
                source[i, :, :] = self.specaug_transform(source[i, :, :])  

        src_lengths=torch.full([source.shape[0]],source.shape[1]) 
        
        if self.feature_grad_mult > 0:
            features, _ = self.subsample(source, src_lengths=src_lengths)
        else:
            with torch.no_grad():
                features, _ = self.subsample(source, src_lengths=src_lengths)
  
        features = features.transpose(0, 1)
        features = features.transpose(1, 2)

        if target_list is not None: 
            features, target_list = self.forward_targets(features, target_list) 

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)  

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)  

        features = self.dropout_input(features)   
        unmasked_features = self.dropout_features(unmasked_features)

        if not self.fine_tuning and mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list) 
        else:
            x = features
            mask_indices = None 

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )  
        
        if self.ils:
            ils_results = []
            for layer in self.predict_layers:
                if layer < len(layer_results):
                    ils_results.append(layer_results[layer][0].transpose(0, 1))
                else:
                    ils_results.append(layer_results[-1][0].transpose(0, 1))

        if features_only:  
            return {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

        if self.ils:
            logit_m_list = []
            logit_u_list = []
            targ_m_list_all = []
            targ_u_list_all = []

            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)

            targ_m_list=target_list[0][masked_indices]
            targ_m_list=targ_m_list.long()
            
            targ_u_list=target_list[0][nomask_indices]
            targ_u_list = targ_u_list.long()
            
            for idx, layer_x in enumerate(ils_results):
                if not self.skip_masked:
                    proj_x_m = self.final_proj[idx](layer_x[masked_indices])
                    proj_x_m /= self.logit_temp
                    logit_m_list.append(proj_x_m )
                else:
                    logit_m_list += [None for _ in target_list]

                if not self.skip_nomask:
                    proj_x_u = self.final_proj[idx](layer_x[nomask_indices])
                    proj_x_u /= self.logit_temp
                    logit_u_list.append(proj_x_u )
                else:
                    logit_u_list += [None for _ in target_list]
                    
                targ_m_list_all.append(targ_m_list)
                targ_u_list_all.append(targ_u_list)
                
        else: 
            if not self.skip_masked:
                masked_indices = torch.logical_and(~padding_mask, mask_indices)
                proj_x_m = self.final_proj(x[masked_indices])
                proj_x_m /= self.logit_temp
                logit_m_list = [proj_x_m for _ in range(len(target_list))]
            else:
                logit_m_list = [None for _ in target_list]

            if not self.skip_nomask:
                nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
                proj_x_u = self.final_proj(x[nomask_indices])
                proj_x_u /= self.logit_temp
                logit_u_list = [proj_x_u for _ in range(len(target_list))]
            else:
                logit_u_list = [None for _ in target_list]

            targ_m_list=target_list[0][masked_indices]
            targ_m_list=targ_m_list.long()
            targ_m_list = [targ_m_list for _ in range(len(target_list))]

            targ_u_list=target_list[0][nomask_indices]
            targ_u_list = targ_u_list.long()
            targ_u_list = [targ_u_list for _ in range(len(target_list))]
        

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask, 
            "targ_m_list": targ_m_list_all,
            "targ_u_list": targ_u_list_all,
        }
        return result
    
