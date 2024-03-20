#!/bin/bash

envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1" "door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")

# crowdsourced labels (CS) + linear (MLP)
domain="adroit"
modality="state"
structure="mlp"
fake_label=false
ensemble_size=3
n_epochs=100
num_query=2000
len_query=50
data_dir="../crowdsource_human_labels"
seed=999
exp_name="CS-MLP"

for env in "${envs[@]}"
do
    nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
        seed=$seed exp_name=$exp_name >/dev/null 2>&1 &
done

envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1" "door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")

# crowdsourced labels (CS) + transformer (TFM)
domain="adroit"
modality="state"
structure="transformer1"
fake_label=false
ensemble_size=3
n_epochs=100
num_query=2000
len_query=50
data_dir="../crowdsource_human_labels"
seed=999
exp_name="CS-TFM"
max_seq_len=50

for env in "${envs[@]}"
do
    nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
        seed=$seed exp_name=$exp_name max_seq_len=$max_seq_len >/dev/null 2>&1 &
done