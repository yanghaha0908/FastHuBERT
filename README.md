# __Fast-HuBERT__
## Intro 
***



## Model 
***
![Alt text](model.png)

## 




## Pre-train a Fast-HuBERT model
***

```
$ python fairseq_cli/hydra_train.py \
    --config-dir /path/to/FastHuBERT/config/pretrain \
    --config-name fasthubert_base_lirbispeech.yaml \
    common.user_dir=/path/to/FastHuBERT \
    task.data=/path/to/data \
    task.label_dir=/path/to/labels \
    task.labels='["phn"]' \
```

## Fine-tune a Fast-HuBERT model with a CTC loss
***

