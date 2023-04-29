import os
import logging
import time
import argparse

from config.config import config
from pretrain_solver import PretrainSolver

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    
    struct_time = time.localtime(time.time())  # 得到结构化时间格式
    now_time = time.strftime("%Y%m%d_%H%M%S", struct_time)
    config.train_log_path = config.train_log_path % now_time
    config.train_save_path = config.train_save_path % now_time
    config.train_tb_path = config.train_tb_path % now_time

    os.makedirs(config.train_log_path, exist_ok=True)
    os.makedirs(config.train_save_path, exist_ok=True)
    os.makedirs(config.train_tb_path, exist_ok=True)
        
    
    logging.basicConfig(filename=os.path.join(config.train_log_path, "pretrain.log"),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p', force=True)
    
    logging.info("--------------------------------------------------PreTrain--------------------------------------------")
    logging.info('USE GPU {}'.format(config.gpu_id))
    
    logging.info(config)
    print(config)
    
    solver = PretrainSolver(config=config, logging=logging)
    solver.bulid()
    solver.train()
    solver.close()
