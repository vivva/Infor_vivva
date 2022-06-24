import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler   #标准化？也就是数据缩放？
from utils.timefeatures import time_features   #时间戳的编码处理

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4  #24x 4=96，为什么最后都要×4呢？，16天？？
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]  #输入到informer编码器中序列的长度
            self.label_len = size[1]    #输入到解码器中start token的序列长度
            self.pred_len = size[2]     #需要预测的序列的长度
        # init
        assert flag in ['train', 'test', 'val']    #assert如果发生异常就说明表达示为假。可以理解表示式返回 值为假 时就会触发异常
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]  #设定是train还是test，val

        # print('vivva_type_train?:')
        # print(self.set_type)

        self.features = features
        self.target = target        # 要预测的目标维度特征
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq    #数据粒度
        
        self.root_path = root_path  #数据集所在路径
        self.data_path = data_path  #数据集文件名称
        self.__read_data__()
        # data_init = self.__read_data__()

    def __read_data__(self):        #初始化时最重要的函数就是_read_data_
        self.scaler = StandardScaler()
        # 加载csv数据为DataFrame
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]  #三种训练集对应的起始位置
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]    #三种训练集对应的结束位置
        # 根据flag是train,val,test来选择加载的数据的起始与结束的位置
        # {'train':0, 'val':1, 'test':2}
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # print('vivva_border:')
        # print(border2)
        # print(border1)

        # 根据预测的类型进一步提取数据
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]  #获取除了时间列以外的特征
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]  #单目标的话


        # 如果需要进行归一化，则通过StandardScaler进行归一化处理
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]   #df data预测目标
            self.scaler.fit(train_data.values)  #标准化
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values   #将数据转化为numpy数组   #否则就等于目标的数组形式
            
        df_stamp = df_raw[['date']][border1:border2]    #取出指定范围内的序列数据的'date'列
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  #时间格式的处理
        #获取原始数据？-------vivva
        # if do_predict:
        # data_init = df_stamp['data']   #但是这个时间应该是所有的时间

        # 通过time_features对date数据进行进一步的处理
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)  #这个是比较重要的一个函数

        self.data_x = data[border1:border2]  #需要使用归一化后的data
        if self.inverse:   #如果反转，data y等于预测目标
            self.data_y = df_data.values[border1:border2]    #不需要归一化后的数据，data y就是预测目标的数组形式
        else:
            self.data_y = data[border1:border2]     #如果self.inverse是False, data_x和data_y就完全相同
        self.data_stamp = data_stamp
        # return data_init
    
    def __getitem__(self, index):  #对数据进行处理，模型输入分别为seq_x, seq_y, seq_x_mark, seq_y_mark
        s_begin = index #起始下标值
        s_end = s_begin + self.seq_len  #输入编码器的序列的结束下标值
        r_begin = s_end - self.label_len    #start token起始下标值
        r_end = r_begin + self.label_len + self.pred_len    #解码器输出的序列结束的下标值

        #获取原始时间

        # if index == int(len-1):
        #     data_init =

        seq_x = self.data_x[s_begin:s_end]   #输入编码器的长度
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)  #【boder1：boder2】，第一个是48到96，，第二个是预测的24，
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] #时间戳
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  #实例化标准化函数
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]  #获取有多少个参数，多少列
            df_data = df_raw[cols_data]   #读取到列表的某几列
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)  #mean和std都是0
            data = self.scaler.transform(df_data.values)  #（data-mean）/std
        else:
            data = df_data.values  #否则就直接等于ndarray

        #处理时间列
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:  #如果反转，直接等于df data，就是没标准化之后的
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]  #看scale，
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]   #self.data_x是否inverse
        if self.inverse:  #如果true
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        #数据一般直接保存在init类的属性中，如self.data self.label
        if size == None:
            self.seq_len = 24*4*4  #时间戳的四维数据特征，96个时间长度
            self.label_len = 24*4   #Start token length of Informer decoder (defaults to 48)
            self.pred_len = 24*4    #Prediction sequence length (defaults to 24)
        else:
            self.seq_len = size[0]  #应该是这个，默认输入应该是96
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale  #数据标准化->去均值的中心化（均值变为0）；方差的规模化（方差变为1）
        self.inverse = inverse  #默认false，好像是数据缩放？
        self.timeenc = timeenc  #还不知道这个是干什么的，timeenc=0
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  #调用tools的标准化函数

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  #读取数据，是一个dataframe？
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols: #一般来说是none
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);  # df_raw.columns可以获取到列名和数据类型，这个整体获取到的是列名字，赋值给cols
            cols.remove(self.target);
            cols.remove('date')

        df_raw = df_raw[['date']+cols+[self.target]]   #构建表格？不是很明白这里的意义

        #数据集划分
        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        # type_map = {'train':0, 'val':1, 'test':2}
        #默认的seq_len为24*4*4=384个，val和train重叠96个时间点的数据，test和val重叠96个数据点的数据，是为了有充足的历史数据?
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]   #set_type表示的是train、val or test
        border2 = border2s[self.set_type]
        #以上是划分数据集

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]  #这个是得到列名
            df_data = df_raw[cols_data]     #这个是得到对应的列的值
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  #num_train = int(len(df_raw)*0.7)
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]  #获取时间列
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  #对时间列数据进行时间的格式化
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:  #如果反转的话，data-y直接等于对应预测目标的列的值
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]  #否则，即等于data_x等于scale或不scale后的data（有原数据和inverse后的数据）
        self.data_stamp = data_stamp
    #虽然data_y和data_x可能是一样的，但应该一个是

    def __getitem__(self, index):  #每一次迭代的数据？
        s_begin = index  #应该是继承torch的dataset，index是构造一个生成整数的index，
       #index是根据下面的def len生成，index是从0开始
        #返回的是一个样本的数据或标签
        print('dataloader_vivva_s_begin_index:')
        print(s_begin)
        s_end = s_begin + self.seq_len  #seq_len默认96，24*4*4？是上面意思呢？
        r_begin = s_end - self.label_len  #start token开始的位置，默认48
        r_end = r_begin + self.label_len + self.pred_len  #默认48+24=72

        seq_x = self.data_x[s_begin:s_end]  #直接是data_x，即每个预测目标的列的原始数据值，96
        if self.inverse: #如果反转的话，就等于data_y
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
            #和x重叠的48个长度+【96，96+24】->对矩阵或数列进行合并，0应该代表的是在行方向上连接吧（上下连接）
        else:
            seq_y = self.data_y[r_begin:r_end] #72
        #mark好像都是时间的编码
        seq_x_mark = self.data_stamp[s_begin:s_end]     #96
        seq_y_mark = self.data_stamp[r_begin:r_end]   #72

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1   #步长是1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=True, timeenc=0, freq='15min', cols=None):
        #把scale改成了false
        #将inverse改为true--vivva
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale   #这个是scale是true
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  #调用scaler了，mean=0，std=1,但是进去就是std=1，mean=0

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len  #border1:2
        border2 = len(df_raw)   #border2:98
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # if self.scale:  #进入
        #     self.scaler.fit(df_data.values)
        #     data = self.scaler.transform(df_data.values) #最后返回标准化之后的，计算train的mean和std
        # else:
        #     data = df_data.values
        data = df_data.values  #vivva

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:  #进入的是这个
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:  #进入
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1   #data_x每个数据集的数据条数-96+1，步长是1

    def inverse_transform(self, data):
        print(self.scaler.inverse_transform(data))
        return self.scaler.inverse_transform(data)
