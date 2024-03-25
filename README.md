### Install SMARTS:
```
pip install -e git+https://github.com/huawei-noah/SMARTS.git@2690423f048cfd01929d4b4491a871073db9070f#egg=smarts
```

### Train reward model (an example)
```
python preference/train_reward_model.py domain=smarts env=left_c modality=state fake_label=false ensemble_size=3 n_epochs=2000 num_query=2000 len_query=50 data_dir=/path/to/label exp_name=exp_name structure=mlp raw_dataset_dir=path/to/raw_data
```

### Train offline algo (an example)
```
python algorithms/offline/iql_p.py dataset_path path/to/raw_data  reward_model_path path/to/reward/model
```