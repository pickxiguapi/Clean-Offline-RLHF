#!/bin/bash
# export WANDB_API_KEY='****'
export WANDB_MODE='online'

################### CS-MLP ######################

# CS-MLP-Pen-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Pen-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/pen-human-v1/CS-MLP/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Pen-cloned-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Pen-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=pen
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/pen-cloned-v1/CS-MLP/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Door-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Door-human-v1
    mkdir -p ./logs/$name
    device="cuda:2"
    env_type=door
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/door-human-v1/CS-MLP/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Door-cloned-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Door-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:3"
    env_type=door
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/door-cloned-v1/CS-MLP/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Hammer-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Hammer-human-v1
    mkdir -p ./logs/$name
    device="cuda:4"
    env_type=hammer
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-human-v1/CS-MLP/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Hammer-cloned-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Hammer-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:7"
    env_type=hammer
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-cloned-v1/CS-MLP/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

################## CS-TFM ######################

# CS-TFM-Pen-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Pen-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/pen-human-v1/CS-TFM/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Pen-cloned-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Pen-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=pen
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/pen-cloned-v1/CS-TFM/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Door-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Door-human-v1
    mkdir -p ./logs/$name
    device="cuda:2"
    env_type=door
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/door-human-v1/CS-TFM/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Door-cloned-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Door-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:3"
    env_type=door
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/door-cloned-v1/CS-TFM/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Hammer-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Hammer-human-v1
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=hammer
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-human-v1/CS-TFM/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Hammer-cloned-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Hammer-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:2"
    env_type=hammer
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-cloned-v1/CS-TFM/epoch_100_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done