import os
import logging
import time

from config.config import config
from finetune_solver import FinetuneSolver

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    
    struct_time = time.localtime(time.time())  # 得到结构化时间格式
    now_time = time.strftime("%Y%m%d_%H%M%S", struct_time)
    config.finetune_log_path = config.finetune_log_path % now_time
    config.finetune_save_path = config.finetune_save_path % now_time
    config.finetune_tb_path = config.finetune_tb_path % now_time

    os.makedirs(config.finetune_log_path, exist_ok=True)
    os.makedirs(config.finetune_save_path, exist_ok=True)
    os.makedirs(config.finetune_tb_path, exist_ok=True)
        
    
    logging.basicConfig(filename=os.path.join(config.finetune_log_path, "finetune.log"),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p', force=True)
    
    logging.info("--------------------------------------------------Finetune--------------------------------------------")
    logging.info('USE GPU {}'.format(config.gpu_id))
    
    logging.info(config)
    print(config)
    
    solver = FinetuneSolver(config=config, logging=logging)
    solver.bulid()
    solver.train()
    solver.close()
