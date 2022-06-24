import torch.nn as nn
import torch
import math

#词编码
class TokenEmbedding(nn.Module):  #对输入的原始数据进行一个1维卷积得到，将输入数据从7 维映射为 512 维
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        # nn.Conv1d对输入序列的每一个时刻的特征进行一维卷积，且这里stride使用默认的1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):   #返回的是类的实例还是子类的实例？
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # 因为Conv1d要求输入是(N, Cin, L) 输出是(N, Cout, L)，所以需要对输入样本维度顺序进行调整
        # https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

#位置编码
class PositionalEmbedding(nn.Module):  #获取序列中元素的先后关系，位置编码
    def __init__(self, d_model, max_len=5000):  #d_model是维数，默认512，要输出为512维度的？
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        #torch.zeros返回一个由标量0填充的张量，5000行512列
        pe = torch.zeros(max_len, d_model).float()  #创建出了5000个位置的编码，但可能并不需要5000个长度的编码
        pe.require_grad = False  #指定返回的tensor是否需要梯度

        #---利用公式对位置进行编码
        position = torch.arange(0, max_len).float().unsqueeze(1)     # 生成维度为[5000, 1]的位置下标向量
        #unsqueeze对数据维度进行扩充，扩展为【5000，1】
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()  # c_in表示有多少个位置，在时间编码中表示每一维时间特征的粒度（h:24, m:4, weekday:7, day:32, month:13）

        w = torch.zeros(c_in, d_model).float()  #【7，512】
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        #上面的逻辑和位置编码是一样的，不一样的就是fixed的是【7，512】，并且加了下面这两行
        self.emb = nn.Embedding(c_in, d_model)   #nn.embedding专门实现词与词向量之间的映射 【几个词，每个词用多少个向量表示】
        self.emb.weight = nn.Parameter(w, requires_grad=False)   #让某些变量在学习的过程中不断的修改其值以达到最优化
        #.weight是随机初始化方式是标准正态分布n[0,1]

    def forward(self, x):
        return self.emb(x).detach() #不进行训练,#阻止反向传播，tensor永远不需要计算其梯度，不具有grad


class TemporalEmbedding(nn.Module):  # month_embed、day_embed、weekday_embed、hour_embed和minute_embed(可选)多个embedding层处理输入的时间戳，将结果相加
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        #不太明白这里的编码含义？
        minute_size = 4;
        hour_size = 24
        weekday_size = 7;
        day_size = 32;
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding  # 选择用什么方法进行emdedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        # 在数据准备阶段，对于时间的处理时若freq=‘h’时'h':['month','day','weekday','hour']
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    # TemporalEmbedding中的embedding层可以使用Pytorch自带的embedding层，再训练参数，也可以使用定义的FixedEmbedding，
    # 它使用位置编码作为embedding的参数，不需要训练参数


class TimeFeatureEmbedding(nn.Module):  # 使用一个全连接层将输入的时间戳映射到512维的embedding
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)  # 全连接层

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    # 最后以上三部分的embedding加起来，就得到了最终的embedding
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)  #词嵌入
        self.position_embedding = PositionalEmbedding(d_model=d_model)   #位置编码
        # 标准化后的时间才会使用TimeFeatureEmbedding，这是一个可学习的时间编码
        #看embed是fixed还是timeF，选择时间编码的函数
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    # 这里x的输入维度应该是[batch_size, seq_len, dim_feature],x_mark的维度应该是[batch_size, seq_len, dim_date]
    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)