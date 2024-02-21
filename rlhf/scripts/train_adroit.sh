#!/bin/bash

# script teacher (ST) + transformer (TFM)
domain="adroit"
env="pen-cloned-v1"
modality="state"
structure="transformer1"
fake_label=true
ensemble_size=3
n_epochs=200
num_query=2000
len_query=50
data_dir="../generated_fake_labels"
seed=999
exp_name="ST-TFM"

python train_reward_model.py domain=$domain env=$env modality=$modality fake_label=$fake_label ensemble_size=$ensemble_size \
n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir seed=$seed exp_name=$exp_name

# script teacher (ST) + transformer (MLP)
domain="adroit"
env="pen-cloned-v1"
modality="state"
structure="mlp"
fake_label=true
ensemble_size=3
n_epochs=200
num_query=2000
len_query=50
data_dir="../generated_fake_labels"
seed=999
exp_name="ST-MLP"

python train_reward_model.py domain=$domain env=$env modality=$modality fake_label=$fake_label ensemble_size=$ensemble_size \
n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir seed=$seed exp_name=$exp_name

# crowdsourced labels (CS) + transformer (TFM)
domain="adroit"
env="pen-cloned-v1"
modality="state"
structure="transformer1"
fake_label=false
ensemble_size=3
n_epochs=200
num_query=2000
len_query=50
data_dir="../crowdsource_human_labels"
seed=999
exp_name="CS-TFM"

python train_reward_model.py domain=$domain env=$env modality=$modality fake_label=$fake_label ensemble_size=$ensemble_size \
n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir seed=$seed exp_name=$exp_name

# crowdsourced labels (CS) + transformer (MLP)
domain="adroit"
env="pen-cloned-v1"
modality="state"
structure="mlp"
fake_label=false
ensemble_size=3
n_epochs=200
num_query=2000
len_query=50
data_dir="../crowdsource_human_labels"
seed=999
exp_name="CS-MLP"

python train_reward_model.py domain=$domain env=$env modality=$modality fake_label=$fake_label ensemble_size=$ensemble_size \
n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir seed=$seed exp_name=$exp_name