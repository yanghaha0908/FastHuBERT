export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=~/fairseq:$PYTHONPATH

python fairseq_cli/hydra_train.py \
-m \
--config-dir /mnt/lustre/sjtu/home/gry10/FastHuBERT/config/finetune \
--config-name base_100h \
common.user_dir=/mnt/lustre/sjtu/home/gry10/FastHuBERT \
task.data=/mnt/lustre/sjtu/home/gry10/fairseq/bpe_manifest \
task.label_dir=/mnt/lustre/sjtu/home/gry10/fairseq/bpe_manifest \
task.labels=["wp"] \
dataset.train_subset=train_clean_100 \
dataset.valid_subset=dev_other \
dataset.num_workers=2 \
dataset.skip_invalid_size_inputs_valid_test=true \
model.w2v_path=/mnt/lustre/sjtu/home/gry10/fairseq/fasthubert_pretrain_ckpt/checkpoint20.pt \
model.mask_prob=0.65 \
checkpoint.save_dir=/mnt/lustre/sjtu/home/gry10/fairseq/debug \
distributed_training.distributed_world_size=1 \
optimization.update_freq=[8] \
optimization.lr=[0.00003] \
optimization.max_update=80000 \
lr_scheduler.warmup_steps=8000 \
lr_scheduler.hold_steps=32000 \
lr_scheduler.decay_steps=40000 \
task._name=fasthubert_pretraining \
dataset.max_tokens=8750 \
task.stats_npz_path=/data/ygr/librispeech/npyfiles/global_cmvn.npy \
criterion._name=fasthubert_ctc_bpe \
criterion.bert_tokenizer=/mnt/lustre/sjtu/home/gry10/fairseq/bpe_txt/librispeech960_vocab2000.json \
dataset.validate_after_updates=20000 \


# conda activate fairseq

# nohup bash fasthubert_finetune.sh > fasthubert_finetune.log &

# pretrain epoch20!