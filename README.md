# CONet: Channel Optimization for Convolutional Neural Networks #

**Exciting News! CONet has been accepted to ICCV 2021 Workshop: Neural Architectures - Past, Present and Future.**

```
@inproceedings{hosseini2021conet,
  title={CONet: Channel Optimization for Convolutional Neural Networks},
  author={Hosseini, Mahdi S and Zhang, Jia Shu and Liu, Zhe Ming and Fu, Andre and Su, Jingxuan and Tuli, Mathieu and Plataniotis, Konstantinos N},
  booktitle={},
  pages={},
  year={}
}
```

**CONet** is a NAS algorithm for optimizing the channel sizes of Convolutional Neural Networks.
CONet expands/shrinks the channels of convolutional layers based on local indicators after a few epochs of training.


<p align="center">
   <img src="https://user-images.githubusercontent.com/43189683/129463034-e6685296-9c53-47d0-965e-011a6f728916.png" >
</p>


### CIFAR10/100 Results ###
Comparison of CONet results optimizing the [DARTS](https://github.com/quark0/darts/) and [ResNet34](https://github.com/KaimingHe/deep-residual-networks) models
on CIFAR10 (left) and CIFAR100 (right).

<p align="center">
  <img src="https://user-images.githubusercontent.com/43189683/129463282-1f009fe5-28cf-4151-9cc0-b1fbfec19a53.png" width="300"  title="Angular" />
  <img src="https://user-images.githubusercontent.com/43189683/129463283-89b7bf22-7517-4004-a676-1fe023943414.png" width="290" /> 
  </p>


### ImageNet Results ###
Comparison of CONet results for optimizing the [DARTS](https://github.com/quark0/darts/) model.
All DARTS models shown use 14 cells. 

The **Delta** and  **Dataset** columns refer to the delta threshold values and datasets used for 
CONet respectively. _Baseline_ refers to the original DARTS model.

| Delta  | Dataset | Params (M)          |  Top-1 Acc (%)  | Top-5 Acc (%)  |
| ------------------ | :---: | :-------------:| :-----:| :----:|
| Baseline   | -   | 4.7 | 73.3 | 91.3
| 0.0075	 | CIFAR10 | 1.8     |   67.2 | 87.6 |
| 0.0005 | CIFAR10     |    4.8 | 74.0  | 91.8 |
| **0.0005** | **CIFAR100**    |    **9.0** |  **76.6** | **93.2**  |

## Table of Contents ##
- [Requirements](#requirements)
- [Usage](#usage)
  * [Training Options](#training-options)
  * [Important Config Options](#important-config-options)
    + [Config: dataset](#config-dataset)
    + [Config: network](#config-network)
    + [Config: full_train_only](#config-full_train_only)
    + [Config: delta_threshold](#config-delta_threshold)
    + [Config: adapt_trials](#config-adapt_trials)
    + [Config: epochs_per_trial](#config-epochs_per_trial)
    + [Config: mapping_condition_threshold](#config-mapping_condition_threshold)
    + [Config: factor scale](#config-factor-scale)
    + [Config: max_epoch](#config-max_epoch)
  * [Setting Channel Sizes](#setting-channel-sizes)
    + [Channel Sizes for ResNet34](#channel-size-resnet34)
    + [Channel Sizes for DARTS/DARTS+](#channel-size-dartsdarts)
  * [Training Outputs](#training-outputs)
    + [XLSX Output](#xlsx-output)
    + [Channel Size Output](#channel-size-output)
    + [Checkpoints](#checkpoints)
  * [Code Structure](#code-structure)
    + [Main Function](#main-function)
    + [Training Functions](#traning-functions)
    + [CONet Scaling Algorithm](#conet-scaling-algorithm)
    + [CONet Model Creation](#conet-model-instantiation)
    + [Helper functions](#helper-functions)

  

### Requirements ###
We use `Python 3.7`

Per [requirements.txt](requirements.txt), the following Python packages are required:
```text
certifi==2020.6.20
cycler==0.10.0
et-xmlfile==1.0.1
future==0.18.2
graphviz==0.14.2
jdcal==1.4.1
kiwisolver==1.2.0
matplotlib==3.3.2
memory-profiler==0.57.0
numpy==1.19.2
openpyxl==3.0.5
pandas==1.1.3
Pillow==8.0.0
pip==18.1
pkg-resources==0.0.0
psutil==5.7.2
ptflops==0.6.2
pyparsing==2.4.7
python-dateutil==2.8.1
pytz ==2020.1
PyYAML==5.3.1
scipy==1.5.2
setuptools==40.8.0
six==1.15.0
torch==1.6.0
torchvision==0.7.0
torchviz==0.0.1
wheel==0.35.1
xlrd==1.2.0
```


### Usage ###

You can run the code by typing
```console
cd CONet
python NAS_main_search.py --...
```
Where `--...` represents the options for training (see below)

By default this will initiate a channel size search **and** an evaluation of the searched model. 

The dataset, model, and hyperparameters can be adjusted by changing the configurations in `CONet/config.yaml`. 
The description and recommended setting for each configuration option can also be found in `config.yaml`. 
We list the important options below.

#### Training Options ####
Please see the training options below. Please note that for ImageNet, 
the --data option must be used to specify the location of the ImageNet dataset as it can no longer
be downloaded through Pytorch.
```console
--root ROOT            # Set root path of project that parents all others:
                         Default = '.'
--data DATA_PATH       # Set the path to the dataset, 
                         Default = '.adas-data'
--output OUTPUT_PATH   # Set the directory for output files,  
                         Default = 'adas_search'
--checkpoint CKPT_PATH # Set the directory for model checkpoints,
                         Default = ".adas-checkpoint"
```
#### Important Config Options ####
In the following sections we list the key configuration options available to the user.
A more comprehensive description of all options can be found in `CONet/config.yaml`.

- General Settings
  - dataset
  - network
  - full_train_only
- Channel Search Settings
  - delta_threshold
  - adapt_trials
  - epochs_per_trial
  - mapping_condition_threshold
  - factor_scale
- Model Evaluation Settings
  - max_epoch
---

 ##### Config: dataset #####
 Currently only the datasets are supported:
- CIFAR10
- CIFAR100
- ImageNet (Must provide dataset on local storage)
---

##### Config: network #####
The following networks are supported:
- ResNet34
- DARTS
- DARTS+

DARTS and DARTS+ can be instantiated with 7, 14, or 20 cells.

---
##### Config: full_train_only #####
Set to false to perform channel search and model evaluation.

Set to true to only **evaluate** a model with the channel sizes specified in `CONet\global_vars.py`.
We will explain how to set channel sizes in `global_vars.py` below.

---
##### Config: delta_threshold #####
Sets the **rank slope** target for shrinking or expanding channel sizes. 

Referred to as "delta" in algorithm 2 in the CONet paper.

This is sensitive to the model space. For DARTS/DARTS+, we recommend values between 0.0025 to 0.01.
For ResNet, recommend values between 0.01 to 0.025.

---
##### Config: adapt_trials #####
Sets the number of trials for channel search. Channel sizes usually converges around 20 to 25 trials.

---
##### Config: epochs_per_trial #####
Sets the number of epochs per trial. 

---
##### Config: mapping_condition_threshold #####
Sets the mapping condition limit. Layers that exceed that limit are forced to shrink.

Referred to as "mu" in algorithm 2 in the CONet paper.

---
##### Config: factor scale #####
Sets the initial percentage to scale channel sizes.

Referred to as "phi" in algorithm 2 in the CONet paper.

---
##### Config: max_epoch #####
Sets the number of epochs to **evaluate** the model. We recommend 250 epochs for CIFAR
datasets and 200 epochs for ImageNet.

---
#### Setting Channel Sizes ####
Channel sizes can be manually set in `CONet\global_vars.py`. The code will use the
channel sizes in `global_vars.py` as the initial channel sizes for channel searching 
and model evaluation. By default, all channels are set to 32 at the beginning of
channel search.

##### Channel Size: ResNet34 #####
The following variables in `global_vars.py` determines the channel sizes when using ResNet34:
- `super1_idx`
- `super2_idx`
- `super3_idx`
- `super4_idx`

Each variable corresponds to a superblock in ResNet. The channel sizes correspond
directly to the **node** size in the Directed Acyclic Graph representation of ResNet34
as outlined in section 2 in the CONet paper.
For example, the first element in `super1_idx` corresponds to size of the first node.

##### Channel Size: DARTS/DARTS+ #####
The following variables in `global_vars.py` determines the channel sizes when using DARTS/DARTS+ models:
- `DARTS_cell_list_7`
- `DARTS_sep_conv_list_7`
- `DARTS_cell_list_14`
- `DARTS_sep_conv_list_14`
- `DARTS_cell_list_20`
- `DARTS_sep_conv_list_20`
- `DARTSPlus_cell_list_7`
- `DARTSPlus_sep_conv_list_7`
- `DARTSPlus_cell_list_14`
- `DARTSPlus_sep_conv_list_14`
- `DARTSPlus_cell_list_20`
- `DARTSPlus_sep_conv_list_20`

We identified the **per cell** channel size constraints within the DAG representations
 of DARTS and DARTS+ models. 
 
Each `{model}_cell_list_{cell_num}` variable contains one value for each independently adjustable
channel size for each cell. Please note that we count the initial stem layers as "cell 0".
An illustration of the unique channel sizes within DARTS and DARTS+ cells can be found in
`CONet\figures\..`.

Please note that each `sep_conv` operator in DARTS\DARTS+ model actually represents two separable
convolutions in a row. This leaves an intermediate adjustable
channel size for all `sep_conv` connections.

The `{model}_sep_conv_list_{cell_num}` variables stores the intermediate channel sizes for each 
`sep_conv` connection in DARTS/DARTS+ cells. 

#### Training Outputs ####
##### XLSX Output #####
Please note that training progress is conveyed through console outputs, where per-epoch statements are outputed to indicate epoch time and train/test loss/accuracy.

Please note also that a per-epoch updating `.xlsx` file is created for every training session. 
This file reports performance metrics of the CNN during training for both channel search and model evaluation.

The location of the output `.xlsx` file depends on the `-root` and `--output` option during training, and naming of the file is determined by the `config.yaml` file's contents.

##### Channel Size Output #####
After completing a channel search trial, the final channel sizes can be found in the following files in the output directory. 

ResNet34: `Trials\adapted_architectures.xlsx`

DARTS/DARTS+: `Trials\cell_architecture.xlsx` and `Trials\sep_conv_architecture.xlsx` 

These `.xlsx` files contains the channel size of the mode for each trial in separate rows. 
To instantiate a model with these channel sizes, copy the values for an entire row into 
a list and set the corresponding variable in `CONet\global_vars.py` to that list.
The variables are outlined in the previous section.

##### Checkpoints #####
Checkpoints are saved to the path specified by the `-root` and `--checkpoint` option. A file or directory may be passed. If a directory path is specified, the filename for the checkpoint defaults to `ckpt.pth`.

#### Code Structure ####
This sections outlines the code structure for this CONet package.

Due to differences between ResNet34 and DARTS/DARTS+ architectures, the code for
performing channel search and model evaluation are kept separately.


For each architecture (ResNet, DARTS, DARTS+), there are three specific files:
   - Training*: Sets up model training for channel search and model evaluation for that architecture
   - CONet Algorithm: Implements algorithm 2 for that specific architecture
   - Model Creation*: Creates a _CONet version_ of that model with adjustable channel sizes

Please note that users can run all architectures through a common **main function** in 
`NAS_main_search.py`. The main function will read the selected architecture from `config.yaml`
and call the appropriate arch-specific functions.

\* Due to similarities between DARTS and DARTS+, their training and model creation codes are combined in 
`train_DARTS.py` and `model.py` respectively.

##### Main Function #####
The main function is in `CONet\NAS_main_search.py`.

Calling the main function will do the following
- Load settings from `config.yaml`
- Builds necessary directories for output, checkpoints, etc.
- Launch channel search (if `full_train_only` is set to false)
- Launch model evaluation


##### Training Functions #####
The training functions for ResNet34 and DARTS/DARTS+ channel search and model evaluation can be found in 
`CONet\train_ResNet.py` and `CONet\train_DARTS.py` respectively.

In both files, the training functions implement the following:
- Instantiate the necessary components for training (e.g. model, optimizer, etc.)
- For channel search:
  - Train model for a few epochs to generate local indicator metrics
  - Pass metrics to CONet scaling algorithm for that architecture
  - Update model with new channel sizes
 - For model evaluation:
   - Fully train model
   - Report relevant metrics (e.g. best acc, etc.)  



##### CONet Scaling Algorithm #####

The implementation of CONet Scaling algorithm for ResNet, DARTS, and DARTS+ can be found in  
`CONet\resnet_scaling_algorithm.py`, `CONet\darts_scaling_algorithm.py`, and `CONet\dartsplus_scaling_algorithm.py`
respectively.

These files implement lines 3-14 of algorithm 2 in the CONet paper for each architecture. 
The scaling algorithm functions implement the following:
- Maps the local indicator metrics from the `.xlsx` outputs to the channel size variables as listed in `CONet\global_vars.py`
- Calculate metrics (lines 3-5 in algorithm 2) for each channel size
- Calculate new channel sizes based on metrics



##### CONet Model Instantiation #####
The model creation function for ResNet and DARTS/DARTS+ can be found in `CONet\models\own_network.py` and `CONet\model.py` respectively.

The model creation functions implement the following:
- Takes in a list of channel size values for that model
- Instantiate a model with the specified channel sizes.

##### Helper Functions #####
The following files contain common helper functions that may be of interest to users
: `CONet\train_help.py`, `CONet\data.py`, `CONet\dataset.py`

`train_help.py` contains functions for:
- Instantiating optimizers/LR schedulers for training
- Common training loop (e.g. given a model, train for **n** epochs)
- Collection of metrics (e.g. accuracy, rank slope, etc.)

`data.py` contains function for:
- Creating dataloaders for various datasets
- Applying common augmentations

`datasets.py` contains function for:
- Loading the ImageNet dataset