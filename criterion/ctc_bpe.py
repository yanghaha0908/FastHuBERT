# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterionConfig,CtcCriterion
from fairseq.tasks import FairseqTask

from tokenizers import Tokenizer
from tokenizers import decoders

@dataclass
class CtcBpeCriterionConfig(CtcCriterionConfig):  
    bert_tokenizer: str = field(default="", metadata={"help": "path of subword tokenizer"})

@register_criterion("fasthubert_ctc_bpe", dataclass=CtcBpeCriterionConfig)
class CtcBpeCriterion(CtcCriterion):
    def __init__(
        self, cfg: CtcBpeCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(cfg,task,rdrop_alpha)
        
        self.bert_tokenizer = Tokenizer.from_file(cfg.bert_tokenizer)
        self.bert_tokenizer.decoder = decoders.WordPiece()

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:  
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]  
                input_lengths = non_padding_mask.long().sum(-1)  
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )
     
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx 
        )
        targets_flat = sample["target"].masked_select(pad_mask) 
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"] 
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,  
                targets_flat,  
                input_lengths, 
                target_lengths,   
                blank=self.blank_idx,  
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )   

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )  

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens  
        logging_output = {
            "loss": utils.item(loss.data),  
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),  
            "sample_size": sample_size,
        }

        if not model.training:  
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu() 
                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],  
                    input_lengths,  
                ): 
                    lp = lp[:inp_l].unsqueeze(0)  

                    decoded = None
                    if self.w2l_decoder is not None:  
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]
            
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )  
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ) 
                    targ_units_arr = targ.tolist() 
             
                    toks = lp.argmax(dim=-1).unique_consecutive() 
                    pred_units_arr = toks[toks != self.blank_idx].tolist() 
  
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr) 
                    
                    # Decode back to normal words to compute WER
                    token_list = targ_units.split(" ")
                    ids = [self.bert_tokenizer.token_to_id(token) for token in token_list] 
                    targ_words = self.bert_tokenizer.decode(ids) 
                    targ_words = targ_words.replace(" ' ","'")
                    targ_words = targ_words.replace("##","")  
                    targ_words = targ_words.split(" ")  
                    
                    pred_units = self.task.target_dictionary.string(pred_units_arr)  
                    if pred_units == "":
                        pred_words_raw = []
                    else:
                        token_list1 = pred_units.split(" ")    
                        ids1 = [self.bert_tokenizer.token_to_id(token) for token in token_list1]
                        ids1 = [element for element in ids1 if element is not None]

                        pred_words = self.bert_tokenizer.decode(ids1)  
                        pred_words = pred_words.replace(" ' ","'")  
                        pred_words = pred_words.replace("##","")      
                        pred_words_raw = pred_words.split(" ")  
                        
                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:  
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output
