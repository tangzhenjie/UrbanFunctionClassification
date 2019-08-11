import time
import numpy as np
import sys
import datetime
import pandas as pd
import os
import threading

date_dict = {}
days_array = list(pd.date_range(start='10/1/2018', end='3/31/2019'))
tag = 0
for element in days_array:
    key = element._short_repr.replace("-", "")
    date_dict[key] = tag
    tag = tag + 1

def tool(start, end, threshold, visit_list):
    result = 0
    visit_start = str2int[visit_list[0]]
    visit_end = str2int[visit_list[-1]]
    if visit_start < start:
        visit_start = start
    if visit_end > end:
        visit_end = end
    value = visit_end - visit_start + 1
    if value >= threshold:
        result = 1
    return result

# 用字典查询代替类型转换，可以减少一部分计算时间

date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):                                                                  # 0-23
    str2int[str(i).zfill(2)] = i                                                     # zfill（）指定长度为2的字符串右对齐，前面填充0

# 访问记录内的时间从2018年10月1日起,共182天
# 将日期按日历排列

for i in range(182):                          # 0-181
    date = datetime.date(day=1, month=10, year=2018) + datetime.timedelta(days=i)     # 2018-10-01    按照每月的时间间隔相加到下月等
    date_int = int(date.__str__().replace("-", ""))                                   # 20181001      先将data类转换为字符串形式，然后处理
    date2position[date_int] = [i % 7, i//7]                                           # {20181001: [0, 0]}      随机打乱顺序(这里有疑问？？？？，为什么i % 7, i//7  就是几周几天)
    datestr2dateint[str(date_int)] = date_int                                         # {'20181001': 20181001}        随机打乱顺序

# print(date2position)
# print(len(datestr2dateint))


def visit2array_old(table):
    strings = table[1]                                                                # 读取table的标签为1的列（这里1列表示所有用户的访问记录时间）
    init = np.zeros((7, 26, 24))                                                      # 一共182天时间,所以可以分为26的周，每周7天，每天24个小时
    for string in strings:                                                            # strings代表所有人的行为访问记录,  string代表一个用户的行为访问记录
        temp = []
        for item in string.split(','):                                                # 循环处理每个用户的访问记录. item代表的是一天不同时间点的行为访问记录
            temp.append([item[0:8], item[9:].split("|")])                             # 数组存储,前8位为年月日, 9位后记录一天中的时间点. 一天用数组保存，存在数组里[[,...]]
        for date, visit_lst in temp:                                                  #  data表示年月日20181203, visit_lst表示时间点的数组['08'] 或者['00', '01']
            # x - 第几周
            # y - 第几天
            # z - 几点钟
            # value - 到访的总次数
            x, y = date2position[datestr2dateint[date]]                               # 第几周第几天
            for visit in visit_lst:                                                   # 统计到访的总人数
                init[x][y][str2int[visit]] += 1                                       # 统计的是一个人在不同时间点---（第几周第几天不同小时点访问的次数）---的总访问次数(总人数)
    return init

def visit2array_old1(table):
    strings = table[1]                                                                # 读取table的标签为1的列（这里1列表示所有用户的访问记录时间）
    init = np.zeros((7, 26, 25))                                                      # 一共182天时间,所以可以分为26的周，每周7天，每天24个小时
    for string in strings:                                                            # strings代表所有人的行为访问记录,  string代表一个用户的行为访问记录
        temp = []
        for item in string.split(','):                                                # 循环处理每个用户的访问记录. item代表的是一天不同时间点的行为访问记录
            temp.append([item[0:8], item[9:].split("|")])                             # 数组存储,前8位为年月日, 9位后记录一天中的时间点. 一天用数组保存，存在数组里[[,...]]
        for date, visit_lst in temp:                                                  #  data表示年月日20181203, visit_lst表示时间点的数组['08'] 或者['00', '01']
            # x - 第几周
            # y - 第几天
            # z - 几点钟
            # value - 到访的总次数
            x, y = date2position[datestr2dateint[date]]
            init[x][y][-1] += 1
            # 第几周第几天
            for visit in visit_lst:                                                   # 统计到访的总人数
                init[x][y][str2int[visit]] += 1                                       # 统计的是一个人在不同时间点---（第几周第几天不同小时点访问的次数）---的总访问次数(总人数)
    return init


def visit2array(table):
    strings = table[1]                                                                # 读取table的标签为1的列（这里1列表示所有用户的访问记录时间）
    init = np.zeros((182, 24, 2))                                                      # 一共182天时间,所以可以分为26的周，每周7天，每天24个小时
    for string in strings:                                                            # strings代表所有人的行为访问记录,  string代表一个用户的行为访问记录
        temp = []  #   ["20181019", ["11", "12"]] 存储一个人的到访记录
        for item in string.split(','):                                                # 循环处理每个用户的访问记录. item代表的是一天不同时间点的行为访问记录
            temp.append([item[0:8], item[9:].split("|")])                             # 数组存储,前8位为年月日, 9位后记录一天中的时间点. 一天用数组保存，存在数组里[[,...]]
        for date, visit_lst in temp:                                                  #  data表示年月日20181203, visit_lst表示时间点的数组['08'] 或者['00', '01']
            # 天索引
            day_index = date_dict[date]
            init[day_index][len(visit_lst) - 1][1] += 1
            # 第几周第几天
            for visit in visit_lst:                                                   # 统计到访的总人数
                init[day_index][str2int[visit]][0] += 1                                       # 统计的是一个人在不同时间点---（第几周第几天不同小时点访问的次数）---的总访问次数(总人数)


    # 删除一些没用的天
    deletekeys = [30, 51, 60, 91, 122, 131, 150, 181]
    init = np.delete(init, deletekeys, axis=0)
    return init

def visit2array_test():
    table = pd.read_csv("../Dataset/test.txt", header=None)
    filenames = [a[0] for a in table.values]  # final_train_visit_5/5/380485_009.txt
    length = len(filenames)  # 100000
    start_time = time.time()
    for index, filename in enumerate(filenames):  #
        table = pd.read_table("../OriginDataset/" + filename, header=None)
        array = visit2array(table)  # 返回的是一个三维数组，（每个数据表示  第几周第几天不同时间点访问的次数）
        np.save("../Dataset/npy/test_visit/" + filename.split("/")[-1].split(".")[0] + ".npy",
                array)  # 将数组array的数据存储在.npy的文件中
        sys.stdout.write('\r>> Processing visit data %d/%d' % (index + 1, length))  # 打印出循环记录的进度
        sys.stdout.flush()  # 刷新缓存
    sys.stdout.write('\n')  # 换行
    print("using time:%.2fs" % (time.time() - start_time))


def visit2array_train():
    table = pd.read_csv("../Dataset/train.txt", header=None)
    filenames = [a[0] for a in table.values]           # train_part/5/380485_009.txt
    length = len(filenames)                                                                                    # 38200
    start_time = time.time()
    for index, filename in enumerate(filenames):                                                               #
        table = pd.read_table("../OriginDataset/"+filename, header=None)
        array = visit2array(table)                                          # 返回的是一个三维数组，（每个数据表示  第几周第几天不同时间点访问的次数）
        np.save("../Dataset/npy/train_visit/"+filename.split("/")[-1].split(".")[0]+".npy", array)   # 将数组array的数据存储在.npy的文件中
        sys.stdout.write('\r>> Processing visit data %d/%d' % (index+1, length))       # 打印出循环记录的进度
        sys.stdout.flush()                                                             # 刷新缓存
    sys.stdout.write('\n')                                                             # 换行
    print("using time:%.2fs" % (time.time()-start_time))                               # 一次的时间


def visit2array_valid():
    table = pd.read_csv("../Dataset/eval.txt", header=None)
    filenames = [a[0] for a in table.values]  # train_part/5/380485_009.txt
    length = len(filenames)
    start_time = time.time()
    for index, filename in enumerate(filenames):  #
        table = pd.read_table("../OriginDataset/" + filename, header=None)
        array = visit2array(table)  # 返回的是一个三维数组，（每个数据表示  第几周第几天不同时间点访问的次数）
        np.save("../Dataset/npy/train_visit/" + filename.split("/")[-1].split(".")[0] + ".npy",
                array)  # 将数组array的数据存储在.npy的文件中
        sys.stdout.write('\r>> Processing visit data %d/%d' % (index + 1, length))  # 打印出循环记录的进度
        sys.stdout.flush()  # 刷新缓存
    sys.stdout.write('\n')  # 换行
    print("using time:%.2fs" % (time.time() - start_time))


if __name__ == '__main__':
    if not os.path.exists("../Dataset/npy/test_visit/"):
        os.makedirs("../Dataset/npy/test_visit/")
    if not os.path.exists("../Dataset/npy/train_visit/"):
        os.makedirs("../Dataset/npy/train_visit/")

    visit2array_train()
    visit2array_test()
    visit2array_valid()

