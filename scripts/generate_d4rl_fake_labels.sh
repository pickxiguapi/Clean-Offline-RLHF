#!/bin/bash

# domain="mujoco"
# num_query=2000
# len_query=200
# seed=42900

# # mujoco
# envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
#       "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
# for env in "${envs[@]}"
# do
#   python fast_track/generate_d4rl_fake_labels.py --domain $domain --env_name "$env" --num_query $num_query --len_query $len_query --seed $seed
#   wait
# done

domain="antmaze"
num_query=2000
len_query=200
seed=42900

# antmaze
envs=("antmaze-umaze-v2" "antmaze-umaze-diverse-v2" "antmaze-medium-play-v2" "antmaze-medium-diverse-v2" "antmaze-large-play-v2" "antmaze-large-diverse-v2")
for env in "${envs[@]}"
do
  python fast_track/generate_d4rl_fake_labels.py --domain $domain --env_name "$env" --num_query $num_query --len_query $len_query --seed $seed
  wait
done


# domain="adroit"
# num_query=2000
# len_query=50  # 50
# seed=42900

# # adroit
# envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1" "door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")
# for env in "${envs[@]}"
# do
#   python fast_track/generate_d4rl_fake_labels.py --domain $domain --env_name "$env" --num_query $num_query --len_query $len_query --seed $seed
#   wait
# done