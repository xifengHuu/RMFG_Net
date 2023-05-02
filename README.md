# RMFG_Net
## How to use the code
1. Clone this repository:
```
git clone https://github.com/xifengHuu/RMFG_Net.git
cd RMFG_Net
```
2. Install the dependencies:
```
pip install -r requirements.txt
```
3. Prepare the dataset
Place the dataset in the /data according to the format below or modify the train_dataset_root and finetune_dataset_root in /config/config.py
```
RMFG_Net
    |———— dataset
    |        |____ CU_Dataset
    |                 |————  train
    |                 |       |———— sub_dir_1
    |                 |       |         |———— Frame
    |                 |       |         |      |———— 0.png
    |                 |       |         |      |———— 1.png
    |                 |       |         |      |———— ...
    |                 |       |         |____ GT
    |                 |       |                |———— 0.png
    |                 |       |                |———— 1.png
    |                 |       |                |———— ...
    |                 |       |———— sub_dir_2
    |                 |       |         |———— Frame
    |                 |       |         |      |———— ...
    |                 |       |         |____ GT
    |                 |       |                |———— ...
    |                 |       |———— ...
    |                 |————  val
    |                 |       |———— same structure to train
    |                 |____  test
    |                         |———— same structure to train
    |———— ...
```
4. Train
Various training parameters, including learning rate and epoch, can be set in /config/config.py. Besides, the training logs, saved model and tensorboard path can also be modified.
```
python pretrain.py --train_epoch 100 --pretrain_lr 1e-5 --train_dataset_root path_to_dataset --train_batchsize 12
```
5. Finetune
Select a well performing model as the object to be finetuned based on the pre-training situation, please modify the train_state_dict in /config/config.py to your real model path.
```
python finetune.py --finetune_epoch 100 --finetune_epoch 1e-5 --finetune_dataset_root path_to_dataset --finetune_batchsize 4 --finetune_time_clips 3
```
6. Test
Please modify the path to the finetuned model.
```
python test.py
```