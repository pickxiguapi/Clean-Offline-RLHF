#!/bin/bash

# CS-MLP-Hopper-medium-expert-v2
# for ((seed=0; seed<1; seed+=1))
# do
#     name=CS-MLP-IQL-Hopper-medium-expert-v2
#     mkdir -p ./logs/$name
#     device="cuda:0"
#     env_type=hopper
#     dataset=medium_expert_v2
#     reward_model_path="./rlhf/reward_model_logs/hopper-medium-expert-v2/CS-MLP/epoch_50_query_2000_len_200_seed_999/models/reward_model.pt"
#     reward_model_type=mlp
#     config_path=./configs/offline/iql/$env_type/$dataset.yaml
#     nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
#     --reward_model_path $reward_model_path --config_path $config_path \
#     --reward_model_type $reward_model_type --seed $seed >./logs/$name/$seed.log 2>&1 &

#     echo "$name $seed training start!"
# done

# CS-TFM-Hopper-medium-expert-v2
for ((seed=0; seed<1; seed+=1))
do
    name=CS-TFM-IQL-Hopper-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-expert-v2/CS-TFM/epoch_50_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done


