# SIFLoc

# Contents

- [SIFLoc Description](#SIFLoc-description)
- [Model Architecture](#model-arrchitecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)

# [SIFLoc Description](#contents)

SIFLoc is a two-stage model which contains self-supervised pre-training stage and supervised learning stage. The aim of SIFLoc is to promote the performance on recognition of protein subcellular localization in immunofluorescence microscopic images.

Paper: SIFLoc: A self-supervised pre-training method for enhancing the recognition of protein subcellular localization in immunofluorescence microscopic images

# [Model architecture](#contents)

The overall network architecture of SIFLoc is shown in original paper.

# [Dataset](#contents)

Original dataset is from Human Protein Atlas (www.proteinatlas.org). After post-processing, we obtain a custom dataset including 4 parts ( [link1](https://bioimagestore.blob.core.windows.net/dataset/hpa dataSet_part1.zip), [link2](https://bioimagestore.blob.core.windows.net/dataset/hpa_dataSet_part2.zip),[ link3](https://bioimagestore.blob.core.windows.net/dataset/hpa_dataSet_part3.zip), [link4](https://bioimagestore.blob.core.windows.net/dataset/hpa_dataSet_part4.zip)) which is about 6.5GB.
- Dataset size: 173,594 color images (512$\times$512), 13,261 bags in 27 classes
- Data format: RGB images.
  - Note: Data will be processed in src/datasets.py
- Directory structure of the dataset:
```markdown
  .hpa
  ├── ENSG00000001167
  │   ├──686_A3_2_blue_red_green.jpg_1.jpg
  │   ├──686_A3_2_blue_red_green.jpg_2.jpg
  │   ├──686_A3_2_blue_red_green.jpg_3.jpg
  │   ├──......
  ├── ENSG00000001630
  ├── ENSG00000002330
  ├── ENSG00000003756
  ├── ......
```

# [Environment Requirements](#contents)

- Hardware (Ascend/GPU)
  - Prepare hardware environment with Ascend or GPU. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick start](#contents)
After installing MindSpore via the official website, you can start training and evaluation as follows:

- run on Ascend
  ```bash
  # standalone pre-training
  bash scripts/run_pretrain.sh
  # standalone training
  bash scripts/run_train.sh
  # standalone evaluation
  bash scripts/run_eval.sh
  ```
Inside scripts, there are some parameter settings that can be adjusted during pre-training/training/evaluation. 

# [Script description](#contents)

## [Script and sample code](#contents)

```markdown
  . SIFLoc
  ├── Readme.md  									# descriptions about SIFLoc
  ├── scripts
  │   ├──run_pretrain.sh     						 # script to pre-train
  │   ├──run_train.sh     						 	# script to train
  │   ├──run_eval.sh     						 		# script to eval
  ├── src
  │   ├──RandAugment     						 # data augmentation polices
  │   ├──callbacks.py    							 # loss callback 
  │   ├──config.py       							   # parameter configuration
  │   ├──datasets.py    	 						 # creating dataset
  │   ├──eval_metrics.py 					  	# evaluation metrics
  │   ├──loss.py         								# contrastive loss and BCE loss
  │   ├──lr_generator.py 						   # learning rate config
  │   ├──config.py      						  	# parameter configuration
  │   ├──network_define_eval.py     		 # evaluation cell
  │   ├──network_define_pretrain.py 	   # pre-train cell
  │   ├──network_define_train.py    		 # train cell
  │   └── resnet.py    								# backbone network
  ├── enhanced.csv       							# labels of hpa dataset
  ├── eval.py       									# evaluation script
  ├── pretrain.py       					  		# pre-training script
  └── train.py       									# training script
```

## [Script parameters](#contents)

Parameters for both pre-training and training can be set in src/config.py

- config for pre-training
  ```python
  # base setting
  "description": "description.",        # description for pre-training   
  "prefix": prefix,											# prefix for pre-training
  "time_prefix": time_prefix,      			# time prefix
  "network": "resnet18",								# network architecture
  "low_dims": 128,											# the dim of last layer's feature
  "use_MLP": True,											# whether use MLP
  # save
  "save_checkpoint": True, 							# whether save ckpt
  "save_checkpoint_epochs": 1,					# save per <num> epochs
  "keep_checkpoint_max": 2,							# save at most <num> ckpt
  # dataset
  "dataset": "hpa",											# dataset name
  "bag_size": 1,												# bag size = 1 for pre-training
  "classes": 27,												# class number
  "num_parallel_workers": 8,						# num_parallel_workers
  # optimizer
  "base_lr": 0.003,											# init learning rate
  "type": "SGD",												# optimizer type
  "momentum": 0.9,											# momentum
  "weight_decay": 5e-4,									# weight decay
  "loss_scale": 1,											# loss scale
  "sigma": 0.1,													# $\tau$
  # trainer
  "batch_size": 32,											# batch size
  "epochs": 100,												# epochs for pre-training
  "lr_schedule": "cosine_lr",						# learning rate schedule
  "lr_mode": "epoch",										# "epoch" or "step"
  "warmup_epoch": 0,										# epochs for warming up
  ```
- config for training

  ```python
  # base setting
  "description": "description.",        # description for pre-training  
  "prefix": prefix,											# prefix for training
  "time_prefix": time_prefix,      			# time prefix
  "network": "resnet18",								# network architecture
  "low_dims": 128,											# ignoring this for training
  "use_MLP": False,											# whether use MLP (False)
  # save
  "save_checkpoint": True,							# whether save ckpt
  "save_checkpoint_epochs": 1,					# save per <num> epochs
  "keep_checkpoint_max": 2,							# save at most <num> ckpt
  # dataset
  "dataset": "hpa",											# dataset name
  "bag_size_for_train": 1,							# bag size = 1 for training 
  "bag_size_for_eval": 20,							# bag size = 20 for evaluation
  "classes": 27,												# class number
  "num_parallel_workers": 8,						# num_parallel_workers
  # optimizer
  "base_lr": 0.0001,										# init learning rate
  "type": "Adam",												# optimizer type
  "beta1": 0.5,													# beta1
  "beta2": 0.999,												# beta2
  "weight_decay": 0,										# weight decay
  "loss_scale": 1,											# loss scale
  # trainer
  "batch_size_for_train": 8,						# batch size for training
  "batch_size_for_eval": 1,							# batch size for evaluation
  "epochs": 20,													# training epochs
  "eval_per_epoch": 1,									# eval per <num> epochs
  "lr_schedule": "cosine_lr",						# learning rate schedule
  "lr_mode": "epoch",										# "epoch" or "step"
  "warmup_epoch": 0,										# epochs for warming up
  ```