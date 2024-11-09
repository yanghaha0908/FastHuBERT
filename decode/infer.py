#!/usr/bin/env python -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import hashlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import torch
import torch.distributed as dist
# from examples.speech_recognition.new.decoders.decoder_config import (
#     DecoderConfig,
#     FlashlightDecoderConfig,
# )
from decoders.decoder_config import (
    DecoderConfig,
    FlashlightDecoderConfig,
)
# from examples.speech_recognition.new.decoders.decoder import Decoder
from decoders.decoder import Decoder
from fairseq import checkpoint_utils, distributed_utils, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.logging.progress_bar import BaseProgressBar
from fairseq.models.fairseq_model import FairseqModel
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

from tokenizers import Tokenizer
from tokenizers import decoders

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"


@dataclass
class DecodingConfig(DecoderConfig, FlashlightDecoderConfig):
    unique_wer_file: bool = field(
        default=False,
        metadata={"help": "If set, use a unique file for storing WER"},
    )
    results_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, write hypothesis and reference sentences into this directory"
        },
    )


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    decoding: DecodingConfig = DecodingConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


class InferenceProcessor:
    cfg: InferConfig

    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.task = tasks.setup_task(cfg.task)  #没啥问题

        models, saved_cfg = self.load_model_ensemble()  #
        self.models = models
        self.saved_cfg = saved_cfg
        self.tgt_dict = self.task.target_dictionary

        self.task.load_dataset(
            self.cfg.dataset.gen_subset,
            task_cfg=saved_cfg.task,
        )  #
        self.generator = Decoder(cfg.decoding, self.tgt_dict)
        self.gen_timer = StopwatchMeter()
        self.wps_meter = TimeMeter()
        self.num_sentences = 0
        self.total_errors = 0
        self.total_length = 0

        self.hypo_words_file = None
        self.hypo_units_file = None
        self.ref_words_file = None
        self.ref_units_file = None

        self.progress_bar = self.build_progress_bar()
        
        #bpe
        self.bert_tokenizer = Tokenizer.from_file("/mnt/cloudstorfs/sjtu_home/guanrou.yang/fairseq/data/librispeech_data/bpe_txt/librispeech960_vocab2000.json")
        self.bert_tokenizer.decoder = decoders.WordPiece()

    def __enter__(self) -> "InferenceProcessor":
        if self.cfg.decoding.results_path is not None:
            self.hypo_words_file = self.get_res_file("hypo.word")
            self.hypo_units_file = self.get_res_file("hypo.units")
            self.ref_words_file = self.get_res_file("ref.word")
            self.ref_units_file = self.get_res_file("ref.units")
        return self

    def __exit__(self, *exc) -> bool:
        if self.cfg.decoding.results_path is not None:
            self.hypo_words_file.close()
            self.hypo_units_file.close()
            self.ref_words_file.close()
            self.ref_units_file.close()
        return False

    def __iter__(self) -> Any:
        for sample in self.progress_bar:
            if not self.cfg.common.cpu:
                sample = utils.move_to_cuda(sample)

            # Happens on the last batch.
            if "net_input" not in sample:
                continue
            yield sample

    def log(self, *args, **kwargs):
        self.progress_bar.log(*args, **kwargs)

    def print(self, *args, **kwargs):
        self.progress_bar.print(*args, **kwargs)

    def get_res_file(self, fname: str) -> None:
        fname = os.path.join(self.cfg.decoding.results_path, fname)
        if self.data_parallel_world_size > 1:
            fname = f"{fname}.{self.data_parallel_rank}"
        return open(fname, "w", buffering=1)

    def merge_shards(self) -> None:
        """Merges all shard files into shard 0, then removes shard suffix."""

        shard_id = self.data_parallel_rank
        num_shards = self.data_parallel_world_size

        if self.data_parallel_world_size > 1:

            def merge_shards_with_root(fname: str) -> None:
                fname = os.path.join(self.cfg.decoding.results_path, fname)
                logger.info("Merging %s on shard %d", fname, shard_id)
                base_fpath = Path(f"{fname}.0")
                with open(base_fpath, "a") as out_file:
                    for s in range(1, num_shards):
                        shard_fpath = Path(f"{fname}.{s}")
                        with open(shard_fpath, "r") as in_file:
                            for line in in_file:
                                out_file.write(line)
                        shard_fpath.unlink()
                shutil.move(f"{fname}.0", fname)

            dist.barrier()  # ensure all shards finished writing
            if shard_id == (0 % num_shards):
                merge_shards_with_root("hypo.word")
            if shard_id == (1 % num_shards):
                merge_shards_with_root("hypo.units")
            if shard_id == (2 % num_shards):
                merge_shards_with_root("ref.word")
            if shard_id == (3 % num_shards):
                merge_shards_with_root("ref.units")
            dist.barrier()

    def optimize_model(self, model: FairseqModel) -> None:
        model.make_generation_fast_()
        if self.cfg.common.fp16:
            model.half()
        if not self.cfg.common.cpu:
            model.cuda()

    def load_model_ensemble(self) -> Tuple[List[FairseqModel], FairseqDataclass]:
        arg_overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path, separator="\\"),
            arg_overrides=arg_overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )
        for model in models:
            self.optimize_model(model)
        return models, saved_cfg

    def get_dataset_itr(self, disable_iterator_cache: bool = False) -> None:
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.gen_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=(sys.maxsize, sys.maxsize),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        ).next_epoch_itr(shuffle=False)

    def build_progress_bar(
        self,
        epoch: Optional[int] = None,
        prefix: Optional[str] = None,
        default_log_format: str = "tqdm",
    ) -> BaseProgressBar:
        return progress_bar.progress_bar(
            iterator=self.get_dataset_itr(),
            log_format=self.cfg.common.log_format,
            log_interval=self.cfg.common.log_interval,
            epoch=epoch,
            prefix=prefix,
            tensorboard_logdir=self.cfg.common.tensorboard_logdir,
            default_log_format=default_log_format,
        )

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    def process_sentence(
        self,
        sample: Dict[str, Any],
        hypo: Dict[str, Any],
        sid: int,
        batch_id: int,
    ) -> Tuple[int, int]:
        speaker = None  # Speaker can't be parsed from dataset.

        if "target_label" in sample:
            toks = sample["target_label"]
        else:
            toks = sample["target"]  #(2,516)
        toks = toks[batch_id, :]  #(516)  #114
        
        if self.task.cfg.labels == ['wp']:
            hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())  #'HIS AB ##OD ##E WHICH HE HAD FIX ##ED AT A BO ##W ##ERY OR COUNTRY SEA ##T AT A SHORT DISTANCE FROM THE CITY JUST AT WHAT IS NOW CALLED DUT ##CH STREET SOON AB ##OUNDED WITH PRO ##O ##F ##S OF HIS IN ##GE ##N ##U ##ITY PAT ##ENT SM ##OKE J ##ACK ##S THAT REQU ##IRED A HORSE TO WORK THEM DUT ##CH O ##VEN ##S THAT RO ##AST ##ED ME ##AT WITHOUT FIRE CAR ##TS THAT WENT BEFORE THE HORSE ##S WE ##ATHER ##CO ##CK ##S THAT TURNED AGAINST THE WIND AND OTHER WRONG HEAD ##ED CONTR ##IV ##ANCES THAT AST ##ON ##ISHED AND CONF ##OUNDED ALL BEH ##OLD ##ERS'
            # Processes hypothesis.
            if hyp_pieces=="":
                pred_words=""
            else:
                token_list1=hyp_pieces.split(" ")   
                ids1=[self.bert_tokenizer.token_to_id(token) for token in token_list1]
                ids1 = [element for element in ids1 if element is not None]

                pred_words = self.bert_tokenizer.decode(ids1)  #'TRAINGSKSOWLED GENERAL BREAT PAIN SIDE TRY WATER F TEM SOU OBSER LET COMING FREE COMING PERS REST TEARS FIND ED CROWDCIENTAINED INTO RECEIVED SPIRIT SOUAS LET CIRCUMST REPIFUPALTH MISTER FEELING MUCH TRUE AB DURING NIGHT SOU SKWAY SURE REST ANN REST SIDEENCHAMES SIGHT AVREWRIED AGAINIDSON DURING TREAT FAVSONCEDIRC GENERALISON HAPPY INCL POL INTELLOL DESIRE SCHOOL BEGANACTUESANCES THIRTY OCC FAITHVERSIXCIISH SHORTBLEOUEVER READYRIED FOREST SIDE ACCORD DIFFIC REQU SENSE BREITTLEENCH MID MONEYUP GARDEN GONE UNCLEEVER'
                pred_words=pred_words.replace(" ' ","'")  
                pred_words=pred_words.replace("##","")     #'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST AT WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENT SMOKE JACKS THAT REQUIRED A HORSE TO WORK THEM DUTCH OVENS THAT ROASTED MEAT WITHOUT FIRE CARTS THAT WENT BEFORE THE HORSES WEATHERCOCKS THAT TURNED AGAINST THE WIND AND OTHER WRONG HEADED CONTRIVANCES THAT ASTONISHED AND CONFOUNDED ALL BEHOLDERS'
            hyp_words=pred_words
                
            # Processes target.
            target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())  #（506，) 数字
            tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu()) #'HIS AB ##OD ##E WHICH HE HAD FIX ##ED AT A BO ##W ##ERY OR COUNTRY SEA ##T AT A SHORT DISTANCE FROM THE CITY JUST AT WHAT IS NOW CALLED DUT ##CH STREET SOON AB ##OUNDED WITH PRO ##O ##F ##S OF HIS IN ##GE ##N ##U ##ITY PAT ##ENT SM ##OKE J ##ACK ##S THAT REQU ##IRED A HORSE TO WORK THEM DUT ##CH O ##VEN ##S THAT RO ##AST ##ED ME ##AT WITHOUT FIRE CAR ##TS THAT WENT BEFORE THE HORSE ##S WE ##ATHER ##CO ##CK ##S THAT TURNED AGAINST THE WIND AND OTHER WRONG HEAD ##ED CONTR ##IV ##ANCES THAT AST ##ON ##ISHED AND CONF ##OUNDED ALL BEH ##OLD ##ERS'

            token_list=tgt_pieces.split(" ") # print(token_list) ['YES', 'HE', 'ANSWERED', 'I', 'COND', '##UCT', 'THE', 'WOR', '##SHIP', 'I', 'AM', 'A', 'PR', '##IEST']
            ids=[self.bert_tokenizer.token_to_id(token) for token in token_list] #print(ids) [633, 83, 1072, 14, 991, 1846, 59, 233, 1673, 14, 273, 6, 143, 1573]
            targ_words = self.bert_tokenizer.decode(ids)  #YES HE ANSWERED I CONDUCT THE WORSHIP I AM A PRIEST
            targ_words=targ_words.replace(" ' ","'")
            targ_words=targ_words.replace("##","")   #开头会存在## 去不掉  'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST AT WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENT SMOKE JACKS THAT REQUIRED A HORSE TO WORK THEM DUTCH OVENS THAT ROASTED MEAT WITHOUT FIRE CARTS THAT WENT BEFORE THE HORSES WEATHERCOCKS THAT TURNED AGAINST THE WIND AND OTHER WRONG HEADED CONTRIVANCES THAT ASTONISHED AND CONFOUNDED ALL BEHOLDERS'
            #targ_words=targ_words.split(" ") # 底下有split  #['THEN', 'TAKING', 'HEART', 'I', 'TOLD', 'MY', 'GUIDE', 'HOW', 'MORE', 'THAN', 'ONCE', 'I', 'HAD', 'FELT', ...]
            tgt_words=targ_words

        elif self.task.cfg.labels == ['ltr']:
        # Processes hypothesis.
            hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())  #  #tgt_dict 长度是32   4+28  那个txt里是28个 {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12, 'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19, 'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26, "'": 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31}
            if "words" in hypo:  # hyp_pieces ： | A T | T H I S | T U R N I N G | P O I N T | O F | H I S T O R Y | T H E I R | M A N I F E S T | T H E M S E L V E S | S I D E | B Y | S I D E | A N D | O F T E N | M I X E D | A N D | E N T A N G L E D | T O G E T H E R | A | M A G N I F I C E N T | M A N I F O L D | V I R G I N | F O R E S T | L I K E | U P G R O W T H | A N D | U P | S T R I V I N G | A | K I N D | O F | T R O P I C A L | T E M P O | I N | T H E | R I V A L R Y | O F | G R O W T H | A N D | A N | E X T R A O R D I N A R Y | D E C A Y | A N D | S E L F | D E S T R U C T I O N | O W I N G | T O | T H E | S A V A G E L Y | O P P O S I N G | A N D | S E E M I N G L Y | E X P L O D I N G | E G O I S M S | W H I C H | S T R I V E | W I T H | O N E | A N O T H E R | F O R | S U N | A N D | L I G H T | A N D | C A N | N O | L O N G E R | A S S I G N | A N Y | L I M I T | R E S T R A I N T | O R | F O R B E A R A N C E | F O R | T H E M S E L V E S | B Y | M E A N S | O F | T H E | H I T H E R T O | E X I S T I N G | M O R A L I T Y | |
                hyp_words = " ".join(hypo["words"]) #AT THIS TURNING POINT OF HISTORY THEIR MANIFEST THEMSELVES SIDE BY SIDE AND OFTEN MIXED AND ENTANGLED TOGETHER A MAGNIFICENT MANIFOLD VIRGIN FOREST LIKE UPGROWTH AND UP STRIVING A KIND OF TROPICAL TEMPO IN THE RIVALRY OF GROWTH AND AN EXTRAORDINARY DECAY AND SELF DESTRUCTION OWING TO THE SAVAGELY OPPOSING AND SEEMINGLY EXPLODING EGOISMS WHICH STRIVE WITH ONE ANOTHER FOR SUN AND LIGHT AND CAN NO LONGER ASSIGN ANY LIMIT RESTRAINT OR FORBEARANCE FOR THEMSELVES BY MEANS OF THE HITHERTO EXISTING MORALITY
            else:
                hyp_words = post_process(hyp_pieces, self.cfg.common_eval.post_process)

            # Processes target.
            target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())  #（506，) 数字
            tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())  #A T | T H I S | T U R N I N G | P O I N T | O F | H I S T O R Y | T H E R E | M A N I F E S T | T H E M S E L V E S | S I D E | B Y | S I D E | A N D | O F T E N | M I X E D | A N D | E N T A N G L E D | T O G E T H E R | A | M A G N I F I C E N T | M A N I F O L D | V I R G I N | F O R E S T | L I K E | U P | G R O W T H | A N D | U P | S T R I V I N G | A | K I N D | O F | T R O P I C A L | T E M P O | I N | T H E | R I V A L R Y | O F | G R O W T H | A N D | A N | E X T R A O R D I N A R Y | D E C A Y | A N D | S E L F | D E S T R U C T I O N | O W I N G | T O | T H E | S A V A G E L Y | O P P O S I N G | A N D | S E E M I N G L Y | E X P L O D I N G | E G O I S M S | W H I C H | S T R I V E | W I T H | O N E | A N O T H E R | F O R | S U N | A N D | L I G H T | A N D | C A N | N O | L O N G E R | A S S I G N | A N Y | L I M I T | R E S T R A I N T | O R | F O R B E A R A N C E | F O R | T H E M S E L V E S | B Y | M E A N S | O F | T H E | H I T H E R T O | E X I S T I N G | M O R A L I T Y |
            tgt_words = post_process(tgt_pieces, self.cfg.common_eval.post_process)  #结果：AT THIS TURNING POINT OF HISTORY THERE MANIFEST THEMSELVES SIDE BY SIDE AND OFTEN MIXED AND ENTANGLED TOGETHER A MAGNIFICENT MANIFOLD VIRGIN FOREST LIKE UP GROWTH AND UP STRIVING A KIND OF TROPICAL TEMPO IN THE RIVALRY OF GROWTH AND AN EXTRAORDINARY DECAY AND SELF DESTRUCTION OWING TO THE SAVAGELY OPPOSING AND SEEMINGLY EXPLODING EGOISMS WHICH STRIVE WITH ONE ANOTHER FOR SUN AND LIGHT AND CAN NO LONGER ASSIGN ANY LIMIT RESTRAINT OR FORBEARANCE FOR THEMSELVES BY MEANS OF THE HITHERTO EXISTING MORALITY    sentence = sentence.replace(" ", "").replace("|", " ").strip()

        elif self.task.cfg.labels == ['phn']:
            hyp_words = self.tgt_dict.string(hypo["tokens"].int().cpu()) 
            
            target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
            tgt_words = self.tgt_dict.string(target_tokens.int().cpu())
        
              
        if self.cfg.decoding.results_path is not None:
            print(f"{hyp_pieces} ({speaker}-{sid})", file=self.hypo_units_file)
            print(f"{hyp_words} ({speaker}-{sid})", file=self.hypo_words_file)
            print(f"{tgt_pieces} ({speaker}-{sid})", file=self.ref_units_file)
            print(f"{tgt_words} ({speaker}-{sid})", file=self.ref_words_file)

        if not self.cfg.common_eval.quiet:
            logger.info(f"HYPO: {hyp_words}")
            logger.info(f"REF: {tgt_words}")
            logger.info("---------------------")

        hyp_words, tgt_words = hyp_words.split(), tgt_words.split() # 没问题 #['HH', 'W', 'AH1', 'T', 'AH0', 'B', 'AW1', 'T', 'DH', 'IH0', 'S', 'S', 'T', 'AO1', 'R', 'IY0', 'R', 'AY1', 'T', 'IH0', 'Z', 'W', 'AA1', 'Z', 'DH', 'AE1', 'T', 'K', 'R', 'EY1', 'Z', 'IY0', 'IH0', 'T', 'S', 'JH', 'AH1', 'S', 'T', 'S', 'OW1', 'M', 'EH1', 'N', 'IY0', 'TH', 'IH1', 'NG', 'Z', 'L', 'AY1', 'K', 'DH', 'AH0', 'DH', 'AH0', 'R', 'UW1', 'T', ...]

        return editdistance.eval(hyp_words, tgt_words), len(tgt_words)

    def process_sample(self, sample: Dict[str, Any]) -> None:
        self.gen_timer.start()
        hypos = self.task.inference_step(
            generator=self.generator,
            models=self.models,
            sample=sample,    #如果是viterbi  hypos：list2 list：1   只有tokens和score   #505,516
        )                       #hypos[0][0]['tokens'].shape  507  hypos[1][0]['tokens'].shape torch.Size([518])
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos) #1025
        self.gen_timer.stop(num_generated_tokens)
        self.wps_meter.update(num_generated_tokens)

        for batch_id, sample_id in enumerate(sample["id"].tolist()):
            errs, length = self.process_sentence(
                sample=sample,
                sid=sample_id,
                batch_id=batch_id,
                hypo=hypos[batch_id][0],
            )
            self.total_errors += errs
            self.total_length += length

        self.log({"wps": round(self.wps_meter.avg)})
        if "nsentences" in sample:
            self.num_sentences += sample["nsentences"]
        else:
            self.num_sentences += sample["id"].numel()

    def log_generation_time(self) -> None:
        logger.info(
            "Processed %d sentences (%d tokens) in %.1fs %.2f "
            "sentences per second, %.2f tokens per second)",
            self.num_sentences,
            self.gen_timer.n,
            self.gen_timer.sum,
            self.num_sentences / (self.gen_timer.sum + 1e-6),
            1.0 / (self.gen_timer.avg + 1e-6),
        )


def parse_wer(wer_file: Path) -> float:
    with open(wer_file, "r") as f:
        return float(f.readline().strip().split(" ")[1])


def get_wer_file(cfg: InferConfig) -> Path:
    """Hashes the decoding parameters to a unique file ID."""
    base_path = "wer"
    if cfg.decoding.results_path is not None:
        base_path = os.path.join(cfg.decoding.results_path, base_path)

    if cfg.decoding.unique_wer_file:
        yaml_str = OmegaConf.to_yaml(cfg.decoding)
        fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
        return Path(f"{base_path}.{fid % 1000000}")
    else:
        return Path(base_path)


def main(cfg: InferConfig) -> float:
    """Entry point for main processing logic.

    Args:
        cfg: The inferance configuration to use.
        wer: Optional shared memory pointer for returning the WER. If not None,
            the final WER value will be written here instead of being returned.

    Returns:
        The final WER if `wer` is None, otherwise None.
    """

    yaml_str, wer_file = OmegaConf.to_yaml(cfg.decoding), get_wer_file(cfg)

    # Validates the provided configuration.
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 4000000
    if not cfg.common.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA not found; set `cpu=True` to run without CUDA")

    logger.info(cfg.common_eval.path)

    with InferenceProcessor(cfg) as processor:   #InferenceProcessor() 有setup_task load_model load dataset等
        for sample in processor:
            processor.process_sample(sample)     #sample{dict:5} 'id' 'net_input' dict

        processor.log_generation_time()

        if cfg.decoding.results_path is not None:
            processor.merge_shards()

        errs_t, leng_t = processor.total_errors, processor.total_length

        if cfg.common.cpu:
            logger.warning("Merging WER requires CUDA.")
        elif processor.data_parallel_world_size > 1:
            stats = torch.LongTensor([errs_t, leng_t]).cuda()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            errs_t, leng_t = stats[0].item(), stats[1].item()

        wer = errs_t * 100.0 / leng_t

        if distributed_utils.is_master(cfg.distributed_training):
            with open(wer_file, "w") as f:
                f.write(
                    (
                        f"WER: {wer}\n"
                        f"err / num_ref_words = {errs_t} / {leng_t}\n\n"
                        f"{yaml_str}"
                    )
                )

        return wer


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    utils.import_user_module(cfg.common)

    # logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

        wer = parse_wer(get_wer_file(cfg))
    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))

    logger.info("Word error rate: %.4f", wer)
    if cfg.is_ax:
        return wer, None

    return wer


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
