from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            #num of encoder layers，默认是2
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)  #DataParallel多个GPU用的
        return model

    def _get_data(self, flag):  # 主要就是根据所选择的数据集加载数据，之后构建DataSet和DataLoader：
        args = self.args  #第一步得到args


        data_dict = {  # 数据集的加载可以按照不同的时间粒度进行构建
            'ETTh1': Dataset_ETT_hour,  # 这个是一个py文件，写了注释
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,  #进行处理之后得到
        }
        Data = data_dict[self.args.data]   #将数据集参数选择对应的数据处理方法，->方法应该可以自己写
        timeenc = 0 if args.embed != 'timeF' else 1 #time features encoding, options:[timeF, fixed, learned]

        if flag == 'test':
            shuffle_flag = False;   #test不打乱
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
            print('test_batchsize_vivva:')
            print(batch_size)
        elif flag == 'pred':    #调用的是这个
            shuffle_flag = False; #pred也不打乱
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred   #返回的应该是一个dataset？该怎么得到初始的时间呢

        else:   #如果是train和val的话
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        data_set = Data(   #各个数据集对应的不同的dataloader中的数据处理函数，data要求传入的是一个dataset
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        #data_loader之后（继承的torch的dataset），用dataloader承接，接下来实例化一个dataloader
        data_loader = DataLoader(  #dataloader类产生batch的训练数据？
            data_set,  #实例化的dataset，应该是data_loader类返回的数据，返回的是seq_x seq_y seq_x_mark seq_y_mark,应该是一条数据
            batch_size=batch_size,  #返回的是一个batch的样本seq...
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)    #是指不够一个batch-size的数据是被扔掉还是继续以一个小的batchsize继续

        # print('exp_informer_getdata_vivva_data_set_type：')
        # print(type(data_set))   #data.data_loader.Dataset_ETT_hour
        # print('exp_informer_getdata_vivva_data_loader_type：') #torch.utils.data.dataloader.DataLoader
        # print(type(data_loader))
        return data_set, data_loader   #getdata返回的是data_set和data_loader

    def _select_optimizer(self):  # 优化器？
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):  # 验证集？
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)

        # plot--vivva，试试可以不可以
        plt.figure(0)
        plt.plot(total_loss, label='total_loss')
        plt.legend()
        plt.savefig('./loss/val_loss.png')

        total_loss = np.average(total_loss)



        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')  # 数据准备阶段的入口函数
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')   #get_data 返回的是data_loader和data_set
        # 生成用于存储checkpoint的文件路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)  # train_loader是训练数据的分组,长度是数据集/batchsize，每个元素相当于一个分组，每个batch之间是没有交集的
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()   #调用的损失函数为MSEloss

        #应该得到train_loader的最后一批数据？

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # automatic-mixed-precision 模型进行轻量化

        # print('exp_informer_vivva_train_epochs')
        # print(self.args.train_epochs)
        for epoch in range(self.args.train_epochs):   #默认为6
            iter_count = 0   #这个是循环每个epoch
            train_loss = []

            self.model.train()  # 声明模型当前处于训练状态，BatchNorm或Dropout都需要开启
            # model是build_model，返回的是一个model，再调用informer的model（models中的）
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): #进行一个batch一个batch的循环
                #因为data_loader返回的是包含seq-x/y  seq_mark_x/y的四个dataset，所以train_loader也有四个
                iter_count += 1   #这个是循环每个batch

                model_optim.zero_grad()

                #通过_process_one_batch对informer模型进行调用，输入参数，再接受模型的预测输出，再返回真实值
                pred, true = self._process_one_batch(  # 进行mask，返回的是outputs, batch_y
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # 损失函数
                loss = criterion(pred, true)
                train_loss.append(loss.item())   #向列表末尾追加元素。



                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    # 想在这里保存iter和loss
                    #

                if self.args.use_amp:  # amp是自动混合精度？提高运算速度
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            # Loss_train = torch.tensor(train_loss)  #tensor存储和变换数据的主要工具，和np中的多维数组非常相似，将list转换为tensoor
            # torch.save(Loss_train, 'E:/project/Informer/loss_train/epoch_{}'.format(iter_count))

            # 画出损失函数的值--vivva,这个是循环完所有batch之后，但只是一个epoch
            plt.figure()
            plt.plot(train_loss,label='train{}_loss'.format(iter_count))
            plt.legend()
            plt.savefig('./loss/train{}_loss.png'.format(epoch))
            # plt.show()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)  # 每个epoch的loss等于所有iter的loss的平均值
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)  #调用tools中的调整学习率函数
            # 可视化loss---vivva,这里只保存了一个epoch_3的，不知道为啥，应该是没进入循环？
            loss_train = torch.tensor(train_loss)
            loss_val = torch.tensor(vali_loss)
            loss_test = torch.tensor(test_loss)
            torch.save(loss_train, 'E:/project/Informer/loss/train_epoch_{}.npy'.format(epoch))
            torch.save(loss_val, 'E:/project/Informer/loss/val_epoch_{}.npy'.format(epoch))
            torch.save(loss_test, 'E:/project/Informer/loss/test_epoch_{}.npy'.format(epoch))

            #这几个都内容没有，不用画
            # plt.figure(2)
            # plt.plot(loss_train,label='loss_train')
            # plt.legend()
            # plt.savefig('./loss/loss_train{}.png'.format(epoch))
            # plt.show()
            #
            # plt.figure(3)
            # plt.plot(loss_test,label='loss_test')
            # plt.legend()
            # plt.savefig('./loss/loss_test{}.png'.format(epoch))
            # plt.show()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model   #返回的是一个模型

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()   #设定预测模式

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        #画出true和pred,好像是因为preds和tures的维度问题
        # plt.figure(4)
        # plt.plot(trues,label='GroundTruth')
        # plt.plot(preds,label='Prediction')
        # plt.legend()
        # plt.savefig('./loss/test_trues_preds.png')
        # plt.show()

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # 可视化--vivva，这个只是一个数字，不用可视化
        # plt.figure(5)
        # plt.plot(mae,label='mae')
        # plt.legend()
        # plt.savefig('./loss/test_mae.png')
        # # plt.show()
        #
        # plt.figure(6)
        # plt.plot(mse,label='mse')
        # plt.legend()
        # plt.savefig('./loss/test_mse.png')
        # plt.show()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))  #将预训练的参数权重加载到新的模型之中

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(  #同样也是调用这个函数，来得到模型的预测
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        # print('exp_informer_vivva_pred_preds.shape:')
        # print(preds.shape)  #'list' object has no attribute 'shape
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        # decoder input (以下的输入序列的构造应该是在填充需要预测的mask序列的值为固定内容0或1)
        if self.args.padding == 0:
            ## 这里的维度[batch_size, pred_len, feature_dim]
            #中括号从外到里，分别是第一维、第二维shape0第一维，shape-1最后一维
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()  # 需要预测的序列端用全1代替
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(
            self.device)  # 将start token部分与需要预测的序列部分合并
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:  #model中也会判断，如果不需要，则不输出，需要是第二维输出
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]   #输入模型的是这个，第一维就是预测的维度
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:  #如果数据缩放的话没调用
            outputs = dataset_object.inverse_transform(outputs)  # 将输出的数据反归一化
            #train的话dataset_object是train——data
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)    #32 24 7或者1？
        # 如果预测是‘MS’的单输出就是在最后一个位置，否则就是全部所有的维度的特征
        # output是有mask的
        #batch_y是真实的数据

        return outputs, batch_y
