import torch
import torch.nn as nn
import numpy as np
from utils.masking import TriangularCausalMask, ProbMask
from math import sqrt


class FullAttention(nn.Module):  # 普通的多头注意力机制
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  #Q
        # print('attn_vivva_queris.shape：')
        # print(queries.shape)

        _, S, _, D = values.shape   #V
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  #einsum是爱因斯坦求和，用于矩阵运算，前面一部分表示计算操作的字符串，后面是操作对象
        #这应该表示矩阵blhe和bshe点积得到bhls
        # print('attn_vivva_scores:')
        # print(scores)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)   #得到的分数与values计算

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):  # 新提出的attention
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor  #Probsparse attn factor概率稀疏因子？
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # Q，K，V为输入的embedding分别乘上一个权重矩阵得到的query、key、value
    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]这里传入的Q，K的维度都是（32，8，96，64）
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  #这一步得到K_expand，是（32，8，96，96，64）的维度的
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q  # 这一步得到的index_sample的维度是（96，25）的维度
        #torch.randint均匀产生在(L_K, (L_Q, sample_k))之间的随机整数张量
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample,
                   :]  # ProbSparse Self-Attention首先对K进行采样，得到K_sample
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()  # M   这一步得到的index_sample的维度是（96，25）的维度
        #torch.matmul是tensor的乘法，transpose相当于矩阵中的转置，行和列互换
        # find the Top_k query with sparisty measurement        找到M值最大的u个qi ，对这Top-u个qi 关于K求score值：#
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  #torch.div()方法将输入的每个元素除以一个常量，然后返回一个新的修改过的张量
        #Q_K是（32，8，25，96）
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)  #这应该是用V的均值来填充下面的吧
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        #context_in（32，8，96，64）；V：（32，8，96，64）；scores：（32，8，25，96）；scores：（32，8，25，96）；L_Q：{int} 96；attn_mask:None
        if self.mask_flag:  #deccoder是false
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)  #mask应该是用在decoder上的
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) #返回的应该是计算后的注意力分数？
        #这一步是对之前的scores进行softmax，经过softmax的矩阵的维度还是（32，8，25，96）

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  #这好像是转换数据类型？
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)  #返回计算的attn  #应该一个是真实值一个是预测值？
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape  #四维数组？（32，96，8，64）transpose过后的都是（32，8，96，64）
        _, L_K, _, _ = keys.shape   #L_K:96 _:64
        # print('attn_vivva_queries.shape:')    #torch.Size([32, 72, 8, 64])  [32, 96, 8, 64])  [32, 48, 8, 64]
        # print(queries.shape)
        # print('attn_vivva_B:')  #32
        # print(B)

        queries = queries.transpose(2, 1)  #transpose转置
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)进行log
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        #至此，前期的log以及queries和keys的准备工作都完成了，下面开始进行prob_QK的计算，调用prob_QK
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # scores_top (32*8*25*96)
        # index (32*8*25)
        #调用prob_QK返回的是Q_K, M_top

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context，调用_get_initial_context，values是原始的编码，L_Q是prob计算得到的，返回的是context
        context = self._get_initial_context(values, L_Q)   #传入的是：原始的编码和96这个长度，返回的是填充V之后的attn？
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)  #index是0-24？估计是，是前25个
        #一个变量的值改变不受另一个的影响？
        return context.transpose(2, 1).contiguous(), attn

# 会先将输入的embedding分别通过线性映射得到query、key、value。？应该是先输入到encoder中吧？之后encoder中会计算注意力？还将输入维度d划分为多头，
# 接着就执行前面定义的attention操作，最后经过一个线性映射得到输出

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)   #如果d_model=512并且采用默认n_heads=8时，d_keys=64
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention    # FullAttention or ProbAttention
        #全连接只针对最后一维特征进行全连接，nn.linear一般用于设置网络的全连接层，输入输出都是512？
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  #全连接层的输入与输出一般都是二维张量为【batchsize，size】
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)   #分头计算之后再合并起来
        self.n_heads = n_heads
        self.mix = mix  #Informer类和InformerStack类中是False

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape   #B:32  L:96 _:512  含义应该是batchsize_length_width吧
        _, S, _ = keys.shape    #S:96
        H = self.n_heads        #H:8  _:512

        #进行多分头，此时传入的Q K V都是（32，96，8，64）维度的，经过transpose过后的都是（32，8，96，64）维度的
        queries = self.query_projection(queries).view(B, L, H, -1)    #变成 32 96 8 64即(32,96,H,d model / H)进行分头
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(  #调用的是attention，应该是用probattention进行计算，返回的是context和attns
            queries,
            keys,
            values,
            attn_mask  #encoder不进行mask，decoder才需要
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()  #维度是（32，8，96，64）   #transpose如果三维矩阵【0，1，2】，则保持不变【1，0，2】则交换1和0轴
        out = out.view(B, L, -1)    #out的维度应该是[batch_size, seq_len, d_values*n_heads]，
        #多头合并？ 变为（32，96，512）

        return self.out_projection(out), attn   #out_projection前向过程结束后张量维度应该是[batch_size, seq_len, d_model]

