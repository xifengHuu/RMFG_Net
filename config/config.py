import argparse

parser = argparse.ArgumentParser()

# optimizer
parser.add_argument('--seed', type=int, default=666, help='seed')
parser.add_argument('--gpu_id', type=str, default="1", help='train use gpu')
parser.add_argument('--pretrain_lr', type=float, default=1e-4)
parser.add_argument('--finetune_lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='decay epoch')
# parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

# train
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--train_dataset_root', type=str, default="./dataset/JDMDataset/")
parser.add_argument('--train_dataset_list', type=str, default=["train"])
parser.add_argument('--train_mean_std', type=list, default=[[0.1042, 0.1042, 0.1042], [0.1553, 0.1553, 0.1553]])
parser.add_argument('--train_batchsize', type=int, default=12)
parser.add_argument('--train_log_path', type=str, default='output/train/%s/logs')
parser.add_argument('--train_save_path', type=str, default='output/train/%s/save_model')
parser.add_argument('--train_tb_path', type=str, default='output/train/%s/tensorboard')

# fintune
parser.add_argument('--train_state_dict', type=str, default="./output/train/20230405_164934/save_model/Pretrain_70.pth")
parser.add_argument('--finetune_epoch', type=int, default=100)
parser.add_argument('--finetune_dataset_root', type=str, default="./dataset/JDMDataset/val/")
parser.add_argument('--finetune_dataset_list', type=str, default=["3", "l_r_3", "r30_3", "t_b_3"])
parser.add_argument('--finetune_mean_std', type=list, default=[[0.0944, 0.0944, 0.0944], [0.1492, 0.1492, 0.1492]])
parser.add_argument('--finetune_batchsize', type=int, default=4)
parser.add_argument('--finetune_time_clips', type=int, default=3)
parser.add_argument('--finetune_log_path', type=str, default='output/finetune/%s/logs')
parser.add_argument('--finetune_save_path', type=str, default='output/finetune/%s/save_model')
parser.add_argument('--finetune_tb_path', type=str, default='output/finetune/%s/tensorboard')

# test
parser.add_argument('--test_dataset_root', type=str, default="./dataset/JDMDataset/test/")
# device1 ["4", "l_r_4", "r30_4", "t_b_4"]
# device2 ["0", "l_r_0", "r30_0", "t_b_0"]
parser.add_argument('--test_dataset_part', type=list, default=["4", "l_r_4", "r30_4", "t_b_4"])
parser.add_argument('--test_mean_std', type=list, default=[[0.1800, 0.1800, 0.1800], [0.1660, 0.1660, 0.1660]])
parser.add_argument('--test_time_clips', type=int, default=3)
parser.add_argument('--test_save_path', type=str, default="output/test/%s/")

# other
parser.add_argument('--size', type=tuple, default=(512, 416))


config = parser.parse_args()
