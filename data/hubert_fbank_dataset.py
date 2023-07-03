# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import torch
import numpy as np
from typing import Any, List, Optional, Union

from fairseq.data.audio.hubert_dataset import load_label,load_label_offset,verify_label_lengths,HubertDataset

logger = logging.getLogger(__name__)

def load_fbank(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()  
        for ind, line in enumerate(f):  
            items = line.strip().split("\t") 
            sz = int(items[2])   
            if min_keep is not None and sz < min_keep: 
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append("/data/ygr/librispeech/"+ items[1][35:] )   #last change   names.append(items[1])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1   
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    ) 
    return root, names, inds, tot, sizes


class FastHubertDataset(HubertDataset):  
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float], 
        pad_list: List[str],
        eos_list: List[str],
        stats_npz_path: str,
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_fbank(  
            manifest_path, max_keep_sample_size, min_keep_sample_size
        ) 
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels: 
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]  
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates): 
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )   

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )   
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )
        
        # global cmvn
        stats = np.load(stats_npz_path, allow_pickle=True).tolist()
        self.mean, self.std = stats["mean"], stats["std"] 

    def get_fbank(self, index):  
        wav_path = self.audio_names[index]
        fbank = np.load(wav_path, allow_pickle=True) 
        fbank = torch.from_numpy(fbank).float() 

        fbank = np.subtract(fbank, self.mean) 
        fbank = np.divide(fbank, self.std) 

        return fbank

    def __getitem__(self, index):
        wav = self.get_fbank(index) 
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels}

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)  
            end = size - diff + start  
        return wav[start:end, :], start  

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None] 
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]  
        audio_sizes = [s.shape[0] for s in audios]  
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else: 
            audio_size = min(min(audio_sizes), int(self.max_sample_size))
        collated_audios, padding_mask, audio_starts = self.collater_audio(   
            audios, audio_size
        )    
        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]  
        targets_list, lengths_list, ntokens_list = self.collater_label( 
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source":  collated_audios, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]), 
            "net_input": net_input,
        } 

        if self.single_target:  
            batch["target_lengths"] = lengths_list[0]  
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]   
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size,80)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)  
        )   
        audio_starts = [0 for _ in audios] 
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0: 
                assert self.pad_audio  
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,80), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts 

