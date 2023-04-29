import os
from PIL import Image
from torch.utils.data import DataLoader
from model.RMFG_Network import RMFGNet
import torch
import numpy as np
from tqdm import tqdm
from config.config import config
import time
from data.dataset import get_test_dataset


class RMFGTest:
    def __init__(self, model_path):
        self.data_root = config.test_dataset_root
        self.test_dataset = config.test_dataset_part
        self.dataloader = {}
        for dst in self.test_dataset:
            self.dataloader[dst] = DataLoader(get_test_dataset(dst), batch_size=1, shuffle=False, num_workers=8)
        self.model = RMFGNet().cuda()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        now_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        self.tag_dir = config.test_save_path % now_time
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.start_time = time.time()
    
    def export(self):
        x = torch.randn((1, 3, 3, 512, 416)).cuda()
        torch.onnx.export(self.model,  # model being run
                            x,  # model input (or a tuple for multiple inputs)
                            "./model.onnx",  # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=12,  # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names=['input'],  # the model's input names
                            output_names=['output']  # the model's output names
                        )

    def test(self):
        with torch.no_grad():
            for dst in self.test_dataset:
                begin_time = round(time.time() * 1000)
                for img, path_li in tqdm(self.dataloader[dst], desc="test:%s" % dst):
                    result = self.model(img.cuda())
                    for res, path in zip(result[:], path_li[:]):
                        save_path = path[0].replace(self.data_root, self.tag_dir).replace(".jpg", ".png").replace('Frame', 'Pred')
                        os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
                        Image.fromarray((res.squeeze().cpu().numpy() * 255).astype(np.uint8)).save(save_path)
                end_time = round(time.time() * 1000)
                print("{} time-consuming: {}ms".format(dst, str(end_time - begin_time)))


if __name__ == "__main__":
    model = RMFGTest("/home/huxf/RMFG_Net/output/finetune/20230405_210248/save_model/Finetune_92.pth")
    model.test()
    
