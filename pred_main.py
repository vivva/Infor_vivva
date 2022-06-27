import os
import argparse
import numpy as np
import sys
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from utils import tools

if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']



args = dotdict()

args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = 'ETTh2' # data
args.root_path = 'data/ETT/' # root path of data file
args.data_path = 'ETTh2.csv' # data file
args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'OT' # target feature in S or MS task
args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = './checkpoints' # location of model checkpoints

args.seq_len = 96 # input sequence length of Informer encoder
args.label_len = 48 # start token length of Informer decoder
args.pred_len = 48 # prediction sequence length,将24改为48试一试
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 7 # encoder input size
args.dec_in = 7 # decoder input size
args.c_out = 7 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0
args.freq = 'h'

args.batch_size = 32
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 1
args.train_epochs = 6
args.patience = 3
args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

args.inverse = True   #vivva



if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
# Set augments by using data name
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    #在这里加入自己的数据集

}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]
print('Args in experiment:')
print(args)
Exp = Exp_Informer  #这里没进入
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)


#设置读取的checkpoint
setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
# prediction = np.load('./results/'+setting+'/real_prediction.npy')  #加载要用到的checkpoint
exp = Exp(args)   #这里进入了，返回的是一个model，这个应该是初始化？
torch.cuda.empty_cache()
# standardscaler = tools.StandardScaler


#获取初始时间，之后可以需要得到多长时间的历史数据，作为历史数据的x
df_raw = pd.read_csv(os.path.join(args.root_path,args.data_path))  #整个csv文件
df_stamp = df_raw[['date']]   #取出指定范围内的序列数据的'date'列
df_stamp['date'] = pd.to_datetime(df_stamp.date)  #时间格式的处理
#作为历史数据的y:df_data
if args.features == 'M':
    cols_data = df_raw.columns[1:]  # 获取除了时间列以外的特征,这里应该是只获取到了列名，.columns只是返回列标签
    #看这部分如何获取到数据的mean和std
    mean = df_raw[[args.target]].mean(0)  #是series？(OT,mean) #mean(0)是代表每一列的平均值,(1)是每一行的均值
    std = df_raw[[args.target]].std(0)
    df_data = df_raw[cols_data][-args.seq_len+1:-1]
elif args.features == 'MS' or args.features == 'S':
    mean = df_raw[[args.target]].mean(0)
    std = df_raw[[args.target]].std(0)
    df_data = df_raw[[args.target]][-args.seq_len+1:-1]

mean = np.array(mean)
std = np.array(std)
print('mean:')
print(mean)
print('std:')
print(std)


def inverse_transform(mean,std,data):
    mean = torch.from_numpy(mean).type_as(data).to(data.device) if torch.is_tensor(data) else mean
    std = torch.from_numpy(std).type_as(data).to(data.device) if torch.is_tensor(data) else std
    return (data * std) + mean

def predict(exp, setting, load=False):
    pred_data, pred_loader = exp._get_data(flag='pred')   #这里用到了exp的get data
    #调用getdata和data_loader
    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))   #加载数据

    exp.model.eval()

    preds = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)  #batch_x/y还都是正常的数
        batch_y = batch_y.float()  #没有batch-y的输入
        # print('batch_x_mark')
        # print(batch_x_mark)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)
        # print(batch_x_mark.shape)   #torch.Size([1, 96, 4])

        # decoder input
        if exp.args.padding == 0:  #0填充
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        elif exp.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        # encoder - decoder
        if exp.args.use_amp:
            with torch.cuda.amp.autocast():
                if exp.args.output_attention:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if exp.args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # print(type(outputs))
        f_dim = -1 if exp.args.features == 'MS' else 0
        batch_y = batch_y[:, -exp.args.pred_len:, f_dim:].to(exp.device)

        pred = outputs.detach().cpu().numpy()  # .squeeze()
        # pred = outputs
        preds.append(pred)

        # inverse_transform = inverse_transform()

    preds = np.array(preds)  #四维数组 (1,1,48,7)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]) #三维数组(1,48,7)
    date_end = preds[0, :, :].tolist()

    # print(preds)
    # print(preds[:,:,-1])

    # result save
    folder_path = './results/pred_test' + '/'  #路径改一下，不知道可以不可以
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)
    #应该是在这里改变加代码进行数据缩放
    #多返回应该batch_x_ark,以便于对时间的提取,如何将x_mark变为以前的样子？可以需要直接在dataset中得到？
    if args.inverse:  # 如果数据缩放的话没调用
        preds = preds[:,:,-1].flatten()
        print(type(preds))  #ndarray
        print(preds.shape)  #(48,)
        preds = inverse_transform(mean, std,preds)
        print(preds)

    return preds,date_end

prediction , date_end= predict(exp, setting, True)


def get_date_list(begin_date,end_date):
    # pred_date = []
    date_list = [x.strftime('%Y-%m-%d %H:%M:%S') for x in list(pd.date_range(start=begin_date, end=end_date,freq='H'))]
    return date_list

#--------------数据缩放-----------






#--------历史数据和预测数据处理-----------



# print(df_stamp['date'])
data_init = df_stamp['date'][int(-args.seq_len)+1:-1]
# data_begin = data_init.tolist()
# print(type(data_begin))  #pandas.core.series.Series,变成了list
data_begin = data_init.tolist()  #转变成list,values是转换成数组

#预测未来的时间长度x:
freq = args.freq
data_time = data_init.iloc[-1]  #获取df的最后一行,
data_time2 = data_init.iloc[-2]  #获取df的最后一行,
myfreq = (data_time - data_time2).seconds   #1h，以时间差向前得到时间  #获取两个样本的时间差
begin_time = data_time
# print(begin_time)
# print(myfreq)
end_time = begin_time + datetime.timedelta(seconds=myfreq*(args.pred_len-1))
print(end_time)
pred_data = get_date_list(begin_time,end_time)
#获取历史数据


# def fit(self, data):
#     self.mean = data.mean(0)
#     self.std = data.std(0)

#将历史数据和预测的数据拼接起来
data_x = data_begin + pred_data  #时间拼接,
data_x = np.array(data_x)   #这个是转换为数组
print('data_x.shape:')
print(data_x.shape)

date_begin = np.array(df_data)[:,-1].tolist()
# date_end = prediction[0,:,:].tolist()
print('date_begin:')
print(date_begin)
print('prediction:')
print(prediction)
date_y = date_begin + prediction.tolist()
# date_y = date_begin.extend(prediction)
date_y = np.array(date_y)
print('date_y.shape:')
print(date_y.shape)
# print(date_y[-1].shape)  #(144,7)

#plot
plt.figure()
# plt.figure(figsize=(6, 7))
plt.plot(data_x,date_y,label = 'prediction')   #0应该是time，应该将这里改为真实的数据集中的时间
plt.vlines(data_begin[-1], 0, 50, 'r', '--', label='pred_start')

plt.xlabel('time',fontsize=11,)
plt.ylabel('prediction',fontsize=11)
plt.xticks(rotation = 23)


plt.savefig('./img/pred_result_h2_inverse_false.png')
plt.show()

'''
def inverse_transform(self, data):
    mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
    std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
    return (data * std) + mean
    
'''