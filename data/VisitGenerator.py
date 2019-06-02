# 从2018年10月1日到2019年3月31日一共182天
# 准备从文件每个文件中提取：第一个数组[182,24]含义是;在这182*24=4368个小时，每个小时内到访该地区的人次数
#                          第二个数组[4368]  含义是：每个维度表示每个人出现次数的人的个数，例如第1维数值表示
#                          半年内到访该地区一次的人数。
# (这里我们就不存储类别了因为与图片类别相同)
import pandas as pd
import os
import numpy as np
from scipy import io

date_dict = {}
days_array = list(pd.date_range(start='10/1/2018', end='3/31/2019'))
tag = 0
for element in days_array:
    key = element._short_repr.replace("-", "")
    date_dict[key] = tag
    tag = tag + 1
def GetStatisticDataArray1FromVisit(train_or_eval_txt_path=None, visit_dataset_dir=None, mat_save_path="", tag="train"):
    """
    example:
     train_or_eval_txt_path = "D:\\pycharm_program\\UrbanFunctionClassification\\data\\train_all.txt"
     visit_dataset_dir = "D:\\competition\\data\\train_visit\\train\\"
     mat_save_path = "D:\\pycharm_program\\UrbanFunctionClassification\\visit_mat_dataset\\train_all_dataset2.mat"
    """

    RESULTS = {}
    with open(train_or_eval_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 获取文件中的文件名
            if tag == "train":
                visit_file_name = line.split("\t")[0].split("\\")[1].split(".")[0]  # !!!!!!!注意不同系统上的\斜线不同
                visit_file_path = visit_dataset_dir + visit_file_name + ".txt"
            elif tag == "test":
                visit_file_name = line.split("\t")[0].split(".")[0]  # !!!!!!!注意不同系统上的\斜线不同
                visit_file_path = visit_dataset_dir + visit_file_name + ".txt"
            if os.path.exists(visit_file_path):
                # 如果存在就执行下面的语句
                with open(visit_file_path, "r") as F:
                    array1 = np.zeros((182, 24), dtype=np.int64)
                    visits = F.readlines()
                    # 生成[182,24]
                    for visit in visits:
                        records = visit.split("\t")[1].split(",")
                        for record in records:
                            # 写入第一个数组并生成第二个数组的索引
                            yearmonthday, hours = record.split("&")
                            array1_first_index = date_dict[yearmonthday]
                            array1_second_indexs = hours.split("|")
                            for second_index in array1_second_indexs:
                                second_index_true = int(second_index)
                                array1[array1_first_index][second_index_true] = array1[array1_first_index][second_index_true] + 1
                    # 然后把element放到最后的结果中结果形式为{"文件名字"：element}
                    RESULTS[visit_file_name] = array1
            else:
                raise ValueError("Invalid filename '%s'." % (visit_file_path))
        io.savemat(mat_save_path, RESULTS)
def GetStatisticDataArray2FromVisit(train_or_eval_txt_path=None, visit_dataset_dir=None, mat_save_path="", tag="train"):
    """
        example:
         train_or_eval_txt_path = "D:\\pycharm_program\\UrbanFunctionClassification\\data\\train_all.txt"
         visit_dataset_dir = "D:\\competition\\data\\train_visit\\train\\"
         mat_save_path = "D:\\pycharm_program\\UrbanFunctionClassification\\visit_mat_dataset\\train_all_dataset2.mat"
    """
    RESULTS = {}
    with open(train_or_eval_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 获取文件中的文件名
            if tag == "train":
                visit_file_name = line.split("\t")[0].split("\\")[1].split(".")[0]  # !!!!!!!注意不同系统上的\斜线不同
                visit_file_path = visit_dataset_dir + visit_file_name + ".txt"
            elif tag == "test":
                visit_file_name = line.split("\t")[0].split(".")[0]  # !!!!!!!注意不同系统上的\斜线不同
                visit_file_path = visit_dataset_dir + visit_file_name + ".txt"
            if os.path.exists(visit_file_path):
                # 如果存在就执行下面的语句
                with open(visit_file_path, "r") as F:
                    array2 = np.zeros(4368, dtype=np.int64)
                    visits = F.readlines()
                    # 生成[182,24]，[4368]数组内容
                    for visit in visits:
                        array2_index = -1
                        records = visit.split("\t")[1].split(",")
                        for record in records:
                            # 写入第一个数组并生成第二个数组的索引
                            yearmonthday, hours = record.split("&")
                            array1_second_indexs = hours.split("|")
                            array2_index = array2_index + len(array1_second_indexs)
                        if array2_index == -1:
                            continue
                        else:
                            array2[array2_index] = array2[array2_index] + 1

                    # 然后把element放到最后的结果中结果形式为{"文件名字"：array2}
                    RESULTS[visit_file_name] = array2
            else:
                raise ValueError("Invalid filename '%s'." % (visit_file_path))
        io.savemat(mat_save_path, RESULTS)

def GetStatisticDataArray12FromVisit(train_or_eval_txt_path=None, visit_dataset_dir=None, mat_save_path="", tag="train"):
    """
            example:
             train_or_eval_txt_path = "D:\\pycharm_program\\UrbanFunctionClassification\\data\\train_all.txt"
             visit_dataset_dir = "D:\\competition\\data\\train_visit\\train\\"
             mat_save_path = "D:\\pycharm_program\\UrbanFunctionClassification\\visit_mat_dataset\\train_all_dataset2.mat"
    """
    RESULTS = {}
    with open(train_or_eval_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 获取文件中的文件名
            if tag == "train":
                visit_file_name = line.split("\t")[0].split("\\")[1].split(".")[0]  # !!!!!!!注意不同系统上的\斜线不同
                visit_file_path = visit_dataset_dir + visit_file_name + ".txt"
            elif tag == "test":
                visit_file_name = line.split("\t")[0].split(".")[0]  # !!!!!!!注意不同系统上的\斜线不同
                visit_file_path = visit_dataset_dir + visit_file_name + ".txt"
            if os.path.exists(visit_file_path):
                # 如果存在就执行下面的语句
                with open(visit_file_path, "r") as F:
                    element = []
                    array1 = np.zeros((182, 24), dtype=np.int64)
                    array2 = np.zeros(4368, dtype=np.int64)
                    visits = F.readlines()
                    # 生成[182,24]，[4368]数组内容
                    for visit in visits:
                        array2_index = -1
                        records = visit.split("\t")[1].split(",")
                        for record in records:
                            # 写入第一个数组并生成第二个数组的索引
                            yearmonthday, hours = record.split("&")
                            array1_first_index = date_dict[yearmonthday]
                            array1_second_indexs = hours.split("|")
                            array2_index = array2_index + len(array1_second_indexs)
                            for second_index in array1_second_indexs:
                                second_index_true = int(second_index)
                                array1[array1_first_index][second_index_true] = array1[array1_first_index][second_index_true] + 1
                        if array2_index == -1:
                            continue
                        else:
                            array2[array2_index] = array2[array2_index] + 1
                    # 每个文件生成一个list[narray[182, 24], narray[4368]]
                    element.append(array1)
                    element.append(array2)

                    # 然后把element放到最后的结果中结果形式为{"文件名字"：element}
                    RESULTS[visit_file_name] = element
            else:
                raise ValueError("Invalid filename '%s'." % (visit_file_path))
        io.savemat(mat_save_path, RESULTS)



#GetVisitDateByTrainOrEvalText()





data1 = io.loadmat("D:\\pycharm_program\\visit_static\\test_array1.mat")
data2 = io.loadmat("D:\\pycharm_program\\visit_static\\test_array2.mat")
result = {}
i = 0
for (k,v) in data1.items():
    element = []
    if i < 3:
        i = i + 1
        continue
    element.append(v)
    element.append(data2[k][0])
    result[k] = element

io.savemat("D:\\pycharm_program\\visit_static\\test_array12.mat",  result)



#x = data.keys()
#x1 = x[0][0]
#x2 = x[0][1]
#i = 0
#GetStatisticDataArray1FromVisit(train_or_eval_txt_path="D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\test.txt", visit_dataset_dir="D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\test_visit\\test\\", mat_save_path="D:\\pycharm_program\\visit_static\\test_array1.mat", tag="test")