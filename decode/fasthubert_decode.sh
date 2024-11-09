export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

for SPLIT in dev_clean dev_other test_clean test_other; do \
    python examples/speech_recognition/new/infer.py \
    --config-dir examples/hubert/config/decode \
    --config-name infer_viterbi \
    common.user_dir=/hpc_stor03/sjtu_home/guanrou.yang/FastHuBERT \
    task.data=/mnt/cloudstorfs/sjtu_home/guanrou.yang/fairseq/data/librispeech_data/bpe_manifest \
    task.normalize=false \
    task._name=fasthubert_pretraining \
    task.labels=["wp"] \
    dataset.max_tokens=6875 \
    dataset.gen_subset=${SPLIT} \
    common_eval.path=/mnt/cloudstorfs/sjtu_home/guanrou.yang/fairseq/fine280/fasthubert_S8.pt \
    common_eval.results_path=/hpc_stor03/sjtu_home/guanrou.yang/results/fasthubert_experiment/for_upload_ckpt_github_20241109/decode_result/${SPLIT} \
    common_eval.quiet=true \
    distributed_training.distributed_world_size=1 \
    +task.stats_npz_path=/data/ygr/librispeech/npyfiles/global_cmvn.npy \

done
