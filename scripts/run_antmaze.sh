#!/bin/bash

# for ((seed=0; seed<3; seed+=1))
# do
#     device="cuda:4"
#     env=antmaze-large-diverse-v2
#     reward_model_path="/home/ubuntu/zhaokai/Offline-RLHF-corl/preference/reward_model_logs/antmaze-large-diverse-v2/vanilla/epoch_50_query_2000_len_200_seed_0/models/reward_model.pt"
#     nohup python iql_p.py --seed $seed --env $env --group $env --reward_model_path $reward_model_path --device $device >output_train_iql_p_antmaze-large-diverse-v2_seed$seed.txt 2>&1 &
# done

# for ((seed=0; seed<3; seed+=1))
# do
#     device="cuda:5"
#     env=antmaze-large-play-v2
#     reward_model_path="/home/ubuntu/zhaokai/Offline-RLHF-corl/preference/reward_model_logs/antmaze-large-play-v2/vanilla/epoch_50_query_2000_len_200_seed_0/models/reward_model.pt"
#     nohup python iql_p.py --seed $seed --env $env --group $env --reward_model_path $reward_model_path --device $device >output_train_iql_p_antmaze-large-play-v2_seed$seed.txt 2>&1 &
# done

# for ((seed=0; seed<3; seed+=1))
# do
#     device="cuda:4"
#     env=antmaze-medium-diverse-v2
#     reward_model_path="/home/ubuntu/zhaokai/Offline-RLHF-corl/preference/reward_model_logs/antmaze-medium-diverse-v2/vanilla/epoch_50_query_2000_len_200_seed_0/models/reward_model.pt"
#     nohup python iql_p.py --seed $seed --env $env --group $env --reward_model_path $reward_model_path --device $device >output_train_iql_p_antmaze-medium-diverse-v2_seed$seed.txt 2>&1 &
# done


# for ((seed=0; seed<3; seed+=1))
# do
#     device="cuda:5"
#     env=antmaze-medium-play-v2
#     reward_model_path="/home/ubuntu/zhaokai/Offline-RLHF-corl/preference/reward_model_logs/antmaze-medium-play-v2/vanilla/epoch_50_query_2000_len_200_seed_0/models/reward_model.pt"
#     nohup python iql_p.py --seed $seed --env $env --group $env --reward_model_path $reward_model_path --device $device >output_train_iql_p_antmaze-medium-play-v2_seed$seed.txt 2>&1 &
# done

# for ((seed=0; seed<3; seed+=1))
# do
#     device="cuda:4"
#     env=antmaze-umaze-diverse-v2
#     reward_model_path="/home/ubuntu/zhaokai/Offline-RLHF-corl/preference/reward_model_logs/antmaze-umaze-diverse-v2/vanilla/epoch_50_query_2000_len_200_seed_0/models/reward_model.pt"
#     nohup python iql_p.py --seed $seed --env $env --group $env --reward_model_path $reward_model_path --device $device >output_train_iql_p_antmaze-umaze-diverse-v2_seed$seed.txt 2>&1 &
# done


# for ((seed=0; seed<3; seed+=1))
# do
#     device="cuda:5"
#     env=antmaze-umaze-v2
#     reward_model_path="/home/ubuntu/zhaokai/Offline-RLHF-corl/preference/reward_model_logs/antmaze-umaze-v2/vanilla/epoch_50_query_2000_len_200_seed_0/models/reward_model.pt"
#     nohup python iql_p.py --seed $seed --env $env --group $env --reward_model_path $reward_model_path --device $device >output_train_iql_p_antmaze-umaze-v2_seed$seed.txt 2>&1 &
# done


device="cuda:7"
env=antmaze-medium-play-v2
seed=0
reward_model_path="/home/ubuntu/dzb/Offline-RLHF/preference/reward_model_logs/antmaze-medium-play-v2/transformer/epoch_100_query_2000_len_200_seed_888/models/reward_model.pt"
nohup python -u ../iql_p.py --seed $seed --env $env --group $env --reward_model_path $reward_model_path --device $device &
