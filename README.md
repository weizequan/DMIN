# Dense Modality Interaction Network for Audio-Visual Event Localization

This repo holds the code for the work presented on TMM [[Paper]](https://ieeexplore.ieee.org/document/9712233) 

# Prerequisites

We provide the implementation in PyTorch for the ease of use.

Install the requirements by runing the following command:

```ruby
pip install -r requirements.txt
```

# Code and Data Preparation

We highly appreciate [@YapengTian](https://github.com/YapengTian/AVE-ECCV18) for the shared features and code.

## Download Features ##

Two kinds of features (i.e., Visual features and Audio features) are required for experiments.

- Visual Features: You can download the VGG visual features from [here](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view).
* Audio Features: You can download the VGG-like audio features from [here](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view).
+ Additional Features: You can download the features of background videos [here](https://drive.google.com/file/d/1I3OtOHJ8G1-v5G2dHIGCfevHQPn-QyLh/view), which are required for the experiments of the weakly-supervised setting.

After downloading the features, please place them into the ```data``` folder. The structure of the ```data```  folder is shown as follows:

```ruby
data
|——audio_features.h5
|——audio_feature_noisy.h5
|——labels.h5
|——labels_noisy.h5
|——mil_labels.h5
|——test_order.h5
|——train_order.h5
|——val_order.h5
|——visual_feature.h5
|——visual_feature_noisy.h5
```
## Download Datasets (Optional) ##

You can download the AVE dataset from the repo [here](https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view).

# Training and testing DMIN in a fully-supervised setting 

Training

```ruby
bash supv_train.sh
# The argument "--snapshot_pref" denotes the path for saving checkpoints and code.
```

Evaluating

```ruby
bash supv_test.sh
```

After training, there will be a checkpoint file whose name contains the accuracy on the test set and the number of epoch.

# Training and testing DMIN in a Weakly-supervised setting

Training

```ruby
bash weak_train.sh
```

Evaluating

```ruby
bash weak_test.sh
```

# Citation

Please cite the following paper if you feel this repo useful to your research

```ruby
@ARTICLE{9712233,
  author={Liu, Shuo and Quan, Weize and Wang, Chaoqun and Liu, Yuan and Liu, Bin and Yan, Dong-Ming},
  journal={IEEE Transactions on Multimedia}, 
  title={Dense Modality Interaction Network for Audio-Visual Event Localization}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2022.3150469}}

```
