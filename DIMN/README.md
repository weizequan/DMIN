## Dense Interaction Modality Network for Audio-Visual Event Localization


# Usage Guide

## Prerequisites

We provide the implementation in PyTorch for the ease of use. 
             
Install the requirements by runing the following command:

```bash
pip install -r requirements.txt
```

### Download Features

Two kinds of features (*i.e.*, Visual features and Audio features) are required for experiments.

- Visual Features: You can download the VGG visual features from [here][Visual_feature].

- Audio Features: You can download the VGG-like audio features from [here][Audio_feature].

- Additional Features: You can download the features of background videos [here][Noisy_visual_feature], which are required for the experiments of the weakly-supervised setting.

After downloading the features, please place them into the `data` folder. The structure of the `data` folder is shown as follows:
```
data
├── audio_feature.h5
├── audio_feature_noisy.h5
├── labels.h5
├── labels_noisy.h5
├── mil_labels.h5
├── test_order.h5
├── train_order.h5
├── val_order.h5
├── visual_feature.h5
└── visual_feature_noisy.h5

```


### Download Datasets (Optional)

You can download the AVE dataset from the repo [here][AVE_dataset].


## Training and testing CMRAN *in a fully-supervised setting*

You can run the following command for training and testing the model.
We evaluate the model on the test set *every epoch* (set by the arg `"eval_freq"` in the `configs/default_config.yaml` file) when training.
```bash
bash supv_train.sh
# The argument "--snapshot_pref" denotes the path for saving checkpoints and code.
```
Evaluating

```bash
bash supv_test.sh
```

After training, there will be a checkpoint file whose name contains the accuracy on the test set and the number of epoch.


## Training and testing CMRAN *in a Weakly-supervised setting*

Similar to training the model in a fully-supervised setting, you can run training and testing using the following commands:

Training
```bash
bash weak_train.sh
```

Evaluating
```bash
bash weak_test.sh
```


[AVE_dataset]:  https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK
[Audio_feature]: https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing
[Visual_feature]: https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing
[Noisy_visual_feature]: https://drive.google.com/file/d/1I3OtOHJ8G1-v5G2dHIGCfevHQPn-QyLh/view?usp=sharing
[YAPENG]: https://github.com/YapengTian/AVE-ECCV18
