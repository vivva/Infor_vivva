import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvLayer(nn.Module):    #输入的x的维度是（32，96，512）经过卷积后的x输出的维度是（32，48，512）,维度减少一半
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,  #一维卷积
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)  #标准化
        self.activation = nn.ELU()      #激活函数
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)     #最大池化

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))   #应该是这里让维度减半？  permute时间数组的维度
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):   #包括一个注意力层和两个卷积层512-2048-512以及两个标准化层和dropout层
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model   #512*4=2048
        self.attention = attention   #是attention layer模型
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  #前一个conv的输出为2048，后一个的输入为2048
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):   #输入的x维度为 32 96 512
        # x [B, L, D] 即batchsize length d
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(   #进入attn.py进行attention的详细计算，返回的是out？（还不太清楚是什么）和attn
            x, x, x,        #三个x分别是query key value
            attn_mask = attn_mask
        )
        #
        x = x + self.dropout(new_x)   # x (32,96,512) ,相当于一个残差网络？
        # nex_x是attn中做过稀疏矩阵相乘的attention，而x是原来的词编码

        #从这里开始，得到是标准化的x和y
        y = x = self.norm1(x)   #x进行标准化得到（标准化的x）和y
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))    #(32,2048,96)
        y = self.dropout(self.conv2(y).transpose(-1,1))     #(32,96,512)

        return self.norm2(x+y), attn


class Encoder(nn.Module):   #Encoder类是将前面定义的Encoder层和Distilling操作组织起来，形成一个Encoder模块
    #蒸馏层比encoderlayer少一个，即最后一个不是蒸馏层
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        #应该是encoder由encoderlayer和convlayer组成
        #attn_layer调用的是encoderlayer中的attn，attnlayer就是一个encoderlayer
        self.attn_layers = nn.ModuleList(attn_layers)   #ModuleList用来用列表的形式保存子模块，以更新梯度
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):  #将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                #zip 将这里面的可迭代对象作为参数，将对象中对应的元素打包成一个元组，即将attn和conv对应的元素打包
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)    #进入conv_layer，这个是蒸馏？
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        #返回的x的维度是（32，48，512）
        return x, attns


class EncoderStack(nn.Module):  # 多个replicas并行执行，不同replicas采用不同长度的embedding，最终得到的结果拼接起来作为输出
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = [];
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s);
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns  # x （32，48，512），然后是informer层