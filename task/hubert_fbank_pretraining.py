# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
from dataclasses import dataclass, field
from fairseq.tasks import register_task

from fairseq.tasks.hubert_pretraining import LabelEncoder,HubertPretrainingConfig,HubertPretrainingTask
from ..data.hubert_fbank_dataset import FastHubertDataset

logger = logging.getLogger(__name__)

@dataclass
class FastHubertPretrainingConfig(HubertPretrainingConfig):  
    stats_npz_path: str = field(default="", metadata={"help": "path to the mean and variance of all fbank features."})
    
@register_task("fasthubert_pretraining", dataclass=FastHubertPretrainingConfig)  
class HubertFbankPretrainingTask(HubertPretrainingTask):  

    cfg: FastHubertPretrainingConfig

    def __init__(
        self,
        cfg: FastHubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

    @classmethod
    def setup_task(
        cls, cfg: FastHubertPretrainingConfig, **kwargs
    ) -> "HubertFbankPretrainingTask":   
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"   
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries 
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels] 

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = FastHubertDataset(  
            manifest,
            sample_rate=self.cfg.sample_rate, 
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            stats_npz_path = self.cfg.stats_npz_path, 
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
        )
