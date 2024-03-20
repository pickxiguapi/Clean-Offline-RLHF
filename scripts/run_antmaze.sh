#!/bin/bash
# export WANDB_API_KEY='***'
export WANDB_MODE='online'

################### CS-MLP ######################

# # CS-MLP-Antmaze-umaze-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-MLP-IQL-Antmaze-umaze-v2
#     mkdir -p ./logs/$name
#     device="cuda:0"
#     env_type=antmaze
#     dataset=umaze_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-umaze-v2/CS-MLP/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=mlp
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# # CS-MLP-Antmaze-umaze-diverse-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-MLP-IQL-Antmaze-umaze-diverse-v2
#     mkdir -p ./logs/$name
#     device="cuda:1"
#     env_type=antmaze
#     dataset=umaze_diverse_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-umaze-diverse-v2/CS-MLP/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=mlp
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# # CS-MLP-Antmaze-medium-play-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-MLP-IQL-Antmaze-medium-play-v2
#     mkdir -p ./logs/$name
#     device="cuda:2"
#     env_type=antmaze
#     dataset=medium_play_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-medium-play-v2/CS-MLP/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=mlp
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# # CS-MLP-Antmaze-medium-diverse-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-MLP-IQL-Antmaze-medium-diverse-v2
#     mkdir -p ./logs/$name
#     device="cuda:3"
#     env_type=antmaze
#     dataset=medium_diverse_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-medium-diverse-v2/CS-MLP/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=mlp
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# # CS-MLP-Antmaze-large-play-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-MLP-IQL-Antmaze-large-play-v2
#     mkdir -p ./logs/$name
#     device="cuda:4"
#     env_type=antmaze
#     dataset=large_play_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-large-play-v2/CS-MLP/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=mlp
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# # CS-MLP-Antmaze-large-diverse-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-MLP-IQL-Antmaze-large-diverse-v2
#     mkdir -p ./logs/$name
#     device="cuda:5"
#     env_type=antmaze
#     dataset=large_diverse_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-large-diverse-v2/CS-MLP/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=mlp
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

################### CS-TFM ######################

# CS-TFM-Antmaze-umaze-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Antmaze-umaze-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=antmaze
    dataset=umaze_v2
    reward_model_path="./rlhf/reward_model_logs/antmaze-umaze-v2/CS-TFM/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# # CS-TFM-Antmaze-umaze-diverse-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-TFM-IQL-Antmaze-umaze-diverse-v2
#     mkdir -p ./logs/$name
#     device="cuda:1"
#     env_type=antmaze
#     dataset=umaze_diverse_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-umaze-diverse-v2/CS-TFM/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=transformer
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# CS-TFM-Antmaze-medium-play-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Antmaze-medium-play-v2
    mkdir -p ./logs/$name
    device="cuda:2"
    env_type=antmaze
    dataset=medium_play_v2
    reward_model_path="./rlhf/reward_model_logs/antmaze-medium-play-v2/CS-TFM/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Antmaze-medium-diverse-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Antmaze-medium-diverse-v2
    mkdir -p ./logs/$name
    device="cuda:3"
    env_type=antmaze
    dataset=medium_diverse_v2
    reward_model_path="./rlhf/reward_model_logs/antmaze-medium-diverse-v2/CS-TFM/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# # CS-TFM-Antmaze-large-play-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-TFM-IQL-Antmaze-large-play-v2
#     mkdir -p ./logs/$name
#     device="cuda:4"
#     env_type=antmaze
#     dataset=large_play_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-large-play-v2/CS-TFM/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=transformer
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# # CS-TFM-Antmaze-large-diverse-v2
# for ((seed=0; seed<3; seed+=1))
# do
#     name=CS-TFM-IQL-Antmaze-large-diverse-v2
#     mkdir -p ./logs/$name
#     device="cuda:5"
#     env_type=antmaze
#     dataset=large_diverse_v2
#     reward_model_path="./rlhf/reward_model_logs/antmaze-large-diverse-v2/CS-TFM/epoch_300_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=transformer
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed --name $name \
#     >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done
