import torch
import  math
max_len = 5000

# position = torch.arange(0, max_len).float().unsqueeze(1)  # 生成维度为[5000, 1]的位置下标向量
# # unsqueeze对数据维度进行扩充
# position1 = torch.arange(0, max_len).float()  #是一个tensor
# # ([0.0000e+00, 1.0000e+00, 2.0000e+00,  ..., 4.9970e+03, 4.9980e+03, 4.9990e+03])
#
# print(position.shape)
# print(position1)

# d_model = 512
# div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
# a = torch.arange(0, d_model, 2).float()
# #torch.arange(start=0, end, step=1, out=None)
# print(a.shape)  #256
# print(div_term.shape)  #256

# c_in = 7
# d_model = 512
#
# w = torch.zeros(c_in, d_model).float()  # 【7，512】
# print(w.shape)


# queries = [[[[1,2][1,2]][[1,2][1,2]]][[[3,2][3,2]][[3,2][3,2]]]]
# print(queries.shape)
# print(type(queries))
#
# B, L_Q, H, D = queries.shape  # 四维数组？
# print(B)
# print(L_Q)
# print(H)
# print(D)
#---------dataset和dataloader
import torch
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self):   #将数据保存在类的属性中，如下面
        self.data = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
        self.label = torch.LongTensor([1,1,0,0])
        print('self.data')
        print(self.data)

    def __getitem__(self,index):  #index是一个索引，这个索引的取值范围是要根据__len__这个返回值确定的
        return self.data[index],self.label[index]

    def __len__(self):
        print('len(self.data)')
        print(len(self.data))
        return len(self.data)  #返回值是4，所以index是0123


mydataset  = MyDataset()
mydataloader = DataLoader(dataset=mydataset, batch_size=1)   #dataloader实例

print('mydataloader')
print(mydataloader)
print('len(mydataloader)')
print(len(mydataloader))
for i,(data,label) in enumerate(mydataloader):
    print(data,label)


