import csv
import torch
import argparse
import torchaudio
import numpy as np
import pandas as pd
import torchaudio.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union
from fairseq.data.audio.audio_utils import _get_torchaudio_fbank,convert_waveform


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]


def extract_fbank_features(
    waveform: torch.FloatTensor,
    sample_rate: int,
    output_path: Optional[Path] = None,
    n_mel_bins: int = 80,
):  
    _waveform, _ = convert_waveform(waveform, sample_rate, to_mono=True)
    # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform * (2 ** 15)
    _waveform = _waveform.numpy()

    features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable fbank feature extraction"
        )

    if output_path is not None:
        np.save(output_path.as_posix(), features)
    return features


def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    

def accum_cmvn_stats(features, mean_sum, square_sum, n_frame):
    # accumulate the mean and square mean stats to compute mean and invstd
    if mean_sum is None:
        mean_sum = features.sum(axis=0)  #(80,)
        square_sum = (features ** 2).sum(axis=0)  #(80,)
        n_frame = features.shape[0]
    else:
        mean_sum += features.sum(axis=0)
        square_sum += (features ** 2).sum(axis=0)
        n_frame += features.shape[0]
    return mean_sum, square_sum, n_frame


def save_cmvn_stats(mean_sum, square_sum, n_frame, dirname):
    mean = mean_sum / n_frame
    var = square_sum / n_frame - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    mean = mean.astype("float32")
    std  = std.astype("float32")
    with open(dirname.joinpath("cmvn.stats"), "w") as f_stats:
        f_stats.write(f"{n_frame}\n")
        for m in mean_sum:
            f_stats.write(f"{m} ")
        f_stats.write("\n")
        for v in square_sum:
            f_stats.write(f"{v} ")
        f_stats.write("\n")
    global_cmvn = {"mean": mean, "std": std}
    np.save(dirname.joinpath("global_cmvn.npy"), global_cmvn)


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True) 
    feature_root = out_root / f"{args.prefix}-fbank80"
    feature_root.mkdir(exist_ok=True)

    sample_ids = []
    id2txt = {}
    npy_paths = {}
    npy_lengths = {}
    mean_sum, square_sum, n_frame = None, None, 0

    with open(args.wavscp, "r") as f_wav:
        for line in tqdm(f_wav.readlines()):
            wavfile = line.split()[0].strip()
            sample_id = Path(wavfile).stem
            
            wav, sample_rate = torchaudio.load(wavfile)
            if sample_rate != args.target_sample_rate:  
                wav = F.resample(wav, sample_rate, args.target_sample_rate)
            sample_rate = args.target_sample_rate

            fbank_features = extract_fbank_features(wav, sample_rate, feature_root/f"{sample_id}.npy")
            mean_sum, square_sum, n_frame = accum_cmvn_stats(fbank_features, mean_sum, square_sum, n_frame)
            
            npy_paths[sample_id] = feature_root/f"{sample_id}.npy"
            audio_len = fbank_features.shape[0]
            npy_lengths[sample_id] = audio_len 
            sent = ' '.join(line.split()[1:]).strip()
            id2txt[sample_id] = sent
            sample_ids.append(sample_id)

        # compute global CMVN
        if args.for_train:
            print ("Computing global cmvn stats ...")
            save_cmvn_stats(mean_sum, square_sum, n_frame, out_root)

        # generate tsv file
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for sample_id in sample_ids:
            manifest["id"].append(sample_id)
            manifest["audio"].append(npy_paths[sample_id])
            manifest["n_frames"].append(npy_lengths[sample_id])
            manifest["tgt_text"].append(id2txt[sample_id])
        save_df_to_tsv(pd.DataFrame.from_dict(manifest), out_root/f"{args.prefix}.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--wavscp", "-w", required=True, type=str)  #info文件路径
    parser.add_argument("--for-train", action="store_true")
    parser.add_argument("--prefix", default="train", required=True, type=str) #哪个subset
    parser.add_argument("--target-sample-rate", default=16000, type=int)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
