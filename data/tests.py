'''
from datetime import datetime, date  # 获取当前时间，datetime以毫秒形式存储时间和格式
now = datetime.now()
print(now)

a = date(2022,2,2)
print(a)

b = datetime.strptime('2022-2-2','%Y-%m-%d')
print(b)

from dateutil.parser import parse
c = parse('2022 9 8 14:23:07')   #用于解析时间
print(c)

import pandas as pd
datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
d = pd.to_datetime(datestrs)
print(d)
'''
from data import data_loader

# root_path = 'E:/project/informer_vivva/data/ETT'
# data_loader.Dataset_ETT_hour(root_path)
# #--------以上和data_loader是可以运行的

import pandas as pd

def get_date_list(begin_date,end_date):
    date_list = [x.strftime('%Y-%m-%d %H:%M:%S') for x in list(pd.date_range(start=begin_date, end=end_date,freq='H'))]
    print(date_list)
    return date_list

freq = '1h'



get_date_list('2022-7-1 12:00:00','2022-7-1 16:00:00')











