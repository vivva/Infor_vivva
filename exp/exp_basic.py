import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args   #初始化参数
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)  #用gpu or cpu？

    def _build_model(self):
        raise NotImplementedError   #尚未实现方法或函数
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass   #pass为了保持程序结构的完整性，一般起占位作用->搭建起框架，后面再实现

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
