import pandas as pd
import numpy as np
from statistics import mean
from BTrees.OIBTree import OIBTree
import re

df = pd.read_csv('Sales1.csv')
#df.to_csv('Sales1.txt',sep='\t',index=False)

class table:
    def __init__(self, arr):
        index = 0
        for item in arr[0]:
            self.dict_col[arr[index]] = index
            index = index+1

def check_if_index(hash_dict, btree_dict, col_dict, col, arr):
    table_col = arr + '_' + col
    if table_col not in col_dict:
        return 'null'
    if col_dict[table_col] == 'hash':
        return hash_dict[table_col]
    if col_dict[table_col] == 'Btree':
        return btree_dict[table_col]


def find_col(arr,s):
    for i in range(len(arr[0])):
        if arr[0][i] == s:
            return arr[:,i]

def find_col_num(arr,s):
    for i in range(len(arr[0])):
        if arr[0][i] == s:
            return i


def sort_by_col(a, col_index):
    a1 = a.T

    col_max = a.shape[-1] - 1

    if col_index < col_max:
        # 两行互换
        a1[col_index] = a1[col_index] + a1[col_max]
        a1[col_max] = a1[col_index] - a1[col_max]
        a1[col_index] = a1[col_index] - a1[col_max]

        a2 = np.lexsort(a1)

        # 因为a1的行交换影响了a（虽然id不同，但是是同一片内存），得到序列结果后再换回来，保持a的纯洁
        a1[col_index] = a1[col_index] + a1[col_max]
        a1[col_max] = a1[col_index] - a1[col_max]
        a1[col_index] = a1[col_index] - a1[col_max]
    else:
        a2 = np.lexsort(a1)

    return a[a2]

# def running_mean(l, N):
#     sum = 0
#     result = list( 0 for x in l)
#
#     for i in range( 0, N ):
#         sum = sum + l[i]
#         result[i] = sum / (i+1)
#
#     for i in range( N, len(l) ):
#         sum = sum - l[i-N] + l[i]
#         result[i] = sum / N
#
#     return result
#
#
# def moving_sum(l, N):
#     sum = 0
#     result = list( 0 for x in l)
#
#     for i in range( 0, N ):
#         sum = sum + l[i]
#         result[i] = sum
#
#     for i in range( N, len(l) ):
#         sum = sum - l[i-N] + l[i]
#         result[i] = sum
#
#     return result
#
def txt_to_matrix(filename):
    file = open(filename)
    lines = file.readlines()
    # print lines
    # ['0.94\t0.81\t...0.62\t\n', ... ,'0.92\t0.86\t...0.62\t\n']形式
    #rows = len(lines)  # 文件行数

#     #datamat = np.array((rows, 7))  # 初始化矩阵
#
#     row = 0
#     record = []
#     for line in lines:
#         line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
#         #print(line)
#         record.append(line)
#         #datamat[row, :] = line[:]
#         #row += 1
#     datamat = np.array(record)
#     return datamat
# datalist = []
#         for i in range(0, len(data)):
#             datalist.append(list(data[i]))
#         k = datalist[0].index(var[3])
#         y = datalist[0].index(var[2])
#         print(k)  # k是类别，要排序的类别，y是要排序的数字
#         print(y)
#         res_table = []
#         res_table.append(datalist[0])
#         temp = (sorted(datalist[1:], key=(lambda x: [x[k]])))
#         for i in range(0, len(temp)):
#             res_table.append(temp[i])
#         a = []  # 存储着结果，按照pricerange对qty的测量
#         for c in range(1, len(res_table)):
#             res_table[c][y] = int(res_table[c][y])
#         b = 1
#         i = 2
#         count = 0
#         a.append([res_table[0][y], res_table[0][k]])
#         while i < len(res_table):
#             sum = res_table[i - 1][y]
#             while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k]:
#                 sum = sum + res_table[i][y]  # 求具体的sum数值
#                 i = i + 1
#             else:
#                 count = i - count -1
#                 avg = (sum / count)
#                 a.append([round(avg, 4), res_table[i - 1][k]])
#                 i = i + 1
#         res_table = np.asarray(a)
#         print(res_table)
#
# datalist = []
#         for i in range(0, len(data)):
#             datalist.append(list(data[i]))
#         k = datalist[0].index(var[3])
#         y = datalist[0].index(var[2])
#         print(k)  # k是类别，要排序的类别，y是要排序的数字
#         print(y)
#         res_table = []
#         res_table.append(datalist[0])
#         temp = (sorted(datalist[1:], key=(lambda x: [x[k]])))
#         for i in range(0, len(temp)):
#             res_table.append(temp[i])
#         a = []  # 存储着结果，按照pricerange对qty的测量
#         for c in range(1, len(res_table)):
#             res_table[c][y] = int(res_table[c][y])
#         b = 1
#         i = 2
#         count = 0
#         a.append([res_table[0][y], res_table[0][k]])
#         while i < len(res_table):
#             sum = res_table[i - 1][y]
#             while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k]:
#                 sum = sum + res_table[i][y]  # 求具体的sum数值
#                 i = i + 1
#             else:
#                 count = i - count -1
#                 avg = (sum / count)
#                 a.append([round(avg, 4), res_table[i - 1][k]])
#                 i = i + 1
#         res_table = np.asarray(a)
#         print(res_table)

    var = 'Avggroup','Sales1','qty','pricerange'
    datalist = []

    for i in range(0, len(data)):
        datalist.append(list(data[i]))
    k = datalist[0].index(var[3])
    y = datalist[0].index(var[2])
    print(k)  # k是类别，要排序的类别，y是要排序的数字
    print(y)
    res_table = []
    res_table.append(datalist[0])
    temp = (sorted(datalist[1:], key=(lambda x: [x[k]])))
    for i in range(0, len(temp)):
        res_table.append(temp[i])
    a = []  # 存储着结果，按照pricerange对qty的测量
    for c in range(1, len(res_table)):
        res_table[c][y] = int(res_table[c][y])
    b = 1
    i = 2
    count = 0
    a.append([res_table[0][y], res_table[0][k]])
    while i < len(res_table):
        sum = res_table[i - 1][y]
        while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k]:
            sum = sum + res_table[i][y]  # 求具体的sum数值
            i = i + 1
        else:
            count = i - count - 1
            avg = (sum / count)
            a.append([round(avg, 4), res_table[i - 1][k]])
            i = i + 1
    res_table = np.asarray(a)
    print(res_table)

# def relop(s,loc):
#     if loc == 0:
#         if re.search(r'=', s):
#             if re.search(r'<=', s):
#                 return 1
#             elif re.search(r'>=', var[2]):
#                 return 2
#             elif re.search(r'!=', var[2]):
#                 return 3
#             else:
#                 return 4
#         if re.search(r'<', s):
#             if re.search(r'<=', s):
#                 return 1
#             else:
#                 return 5
#         if re.search(r'>', s):
#             if re.search(r'>=', s):
#                 return 2
#             else:
#                 return 6
#     else:
#         if re.search(r'=', s):
#             if re.search(r'<=', s):
#                 return 2
#             elif re.search(r'>=', var[2]):
#                 return 1
#             elif re.search(r'!=', var[2]):
#                 return 3
#             else:
#                 return 4
#         if re.search(r'<', s):
#             if re.search(r'<=', s):
#                 return 2
#             else:
#                 return 6
#         if re.search(r'>', s):
#             if re.search(r'>=', s):
#                 return 1
#             else:
#                 return 5

def parse_terms(terms):
    if re.search('[a-z]',terms[0]):
        loc = 0
        constant = terms[1]
        if re.search(r'[\+\-\*\/]+', terms[0]):
            item = re.split(r'[\+\-\*\/]+', terms[0])
            col = item[0]
            const = item[1]
            if re.search(r'\+', terms[0]):
                arithop = 1
            if re.search(r'\-', terms[0]):
                arithop = 2
            if re.search(r'\*', terms[0]):
                arithop = 3
            if re.search(r'\/', terms[0]):
                arithop = 4
        else:
            col = terms[0]
            const = 0
            arithop = 1
    if re.search('[a-z]',terms[1]):
        loc = 1
        constant = terms[0]
        if re.search(r'[\+\-\*\/]+', terms[1]):
            item = re.split(r'[\+\-\*\/]+', terms[1])
            col = item[0]
            const = item[1]
            if re.search(r'\+', terms[1]):
                arithop = 1
            if re.search(r'\-', terms[1]):
                arithop = 2
            if re.search(r'\*', terms[1]):
                arithop = 3
            if re.search(r'\/', terms[1]):
                arithop = 4
        else:
            col = terms[1]
            const = 0
            arithop = 1
    return loc, col, const, arithop, constant



def parse_condition(ori_table, s,res,data):
    terms = re.split(r'[\=\>\<\!]+', s)
    loc, col, const, arithop, constant = parse_terms(terms)
    op = relop(var[2], loc)
    print(op)
    # if op is <=
    if op == 1:
        col_num = find_col_num(data, col)
        for i in range(1, len(data)):
            if arithop == 1:
                if float(data[i][col_num]) + int(const) <= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 2:
                if int(data[i][col_num]) - int(const) <= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 3:
                if int(data[i][col_num]) * int(const) <= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 4:
                if int(data[i][col_num]) / int(const) <= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
    # if op is >=
    if op == 2:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            if arithop == 1:
                if int(data[i][col_num]) + int(const) >= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 2:
                if int(data[i][col_num]) - int(const) >= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 3:
                if int(data[i][col_num]) * int(const) >= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 4:
                if int(data[i][col_num]) / int(const) >= int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
    # if op is !=
    if op == 3:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            if arithop == 1:
                if int(data[i][col_num]) + int(const) != int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 2:
                if int(data[i][col_num]) - int(const) != int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 3:
                if int(data[i][col_num]) * int(const) != int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 4:
                if int(data[i][col_num]) / int(const) != int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
    # if op is =
    if op == 4:
        # index
        tab_col = ori_table + '_' + col
        print(tab_col)
        if tab_col in col_dict.keys():
            if col_dict[tab_col] == 'hash':
                temp_dict = hash_index_dict[tab_col]
                # ??????? how to define
                row = temp_dict[constant]
                print(row)
                res.append(list(data[row]))
                print(res)
            if col_dict[tab_col] == 'Btree':
                temp_dict = Btree_index_dict[tab_col]
                # ??????? how to define
                row = temp_dict[constant]
                print(row)
                res.append(list(data[row]))
                print(res)
        # no index
        else:
            # identify column
            col_num = find_col_num(data, terms[0])
            for i in range(0, len(data)):
                if arithop == 1:
                    if int(data[i][col_num]) + int(const) == int(constant):
                        if list(data[i]) not in res:
                            res.append(list(data[i]))
                if arithop == 2:
                    if int(data[i][col_num]) - int(const) == int(constant):
                        if list(data[i]) not in res:
                            res.append(list(data[i]))
                if arithop == 3:
                    if int(data[i][col_num]) * int(const) == int(constant):
                        if list(data[i]) not in res:
                            res.append(list(data[i]))
                if arithop == 4:
                    if int(data[i][col_num]) / int(const) == int(constant):
                        if list(data[i]) not in res:
                            res.append(list(data[i]))
    # if op is <
    if op == 5:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            if arithop == 1:
                if int(data[i][col_num]) + int(const) < int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 2:
                if int(data[i][col_num]) - int(const) < int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 3:
                if int(data[i][col_num]) * int(const) < int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 4:
                if int(data[i][col_num]) / int(const) < int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
    # if op is >
    if op == 6:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            if arithop == 1:
                if int(data[i][col_num]) + int(const) > int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 2:
                if int(data[i][col_num]) - int(const) > int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 3:
                if int(data[i][col_num]) * int(const) > int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))
            if arithop == 4:
                if int(data[i][col_num]) / int(const) > int(constant):
                    if list(data[i]) not in res:
                        res.append(list(data[i]))

    datamat = np.array(res)
    return datamat

def parse_join_terms(var1, var2):
    if re.search(r'[\+\-\*\/]+', var1):
        table1 = var1.split('.')[0]
        temp = var1.split('.')[1]
        item = re.split(r'[\+\-\*\/]+', temp)
        col1 = item[0]
        const1 = item[1]
        if re.search(r'\+', temp):
            arithop1 = 1
        if re.search(r'\-', temp):
            arithop1 = 2
        if re.search(r'\*', temp):
            arithop1 = 3
        if re.search(r'\/', temp):
            arithop1 = 4
    if not (re.search(r'[\+\-\*\/]+', var1)):
        table1 = var1.split('.')[0]
        col1 = var1.split('.')[1]
        const1 = 0
        arithop1 = 5
    if re.search(r'[\+\-\*\/]+', var2):
        table2 = var2.split('.')[0]
        temp = var2.split('.')[1]
        item = re.split(r'[\+\-\*\/]+', temp)
        col2 = item[0]
        const2 = item[1]
        if re.search(r'\+', temp):
            arithop2 = 1
        if re.search(r'\-', temp):
            arithop2 = 2
        if re.search(r'\*', temp):
            arithop2 = 3
        if re.search(r'\/', temp):
            arithop2 = 4
    if not (re.search(r'[\+\-\*\/]+', var2)):
        table2 = var2.split('.')[0]
        col2 = var2.split('.')[1]
        const2 = 0
        arithop2 = 5

    return table1, col1, const1, arithop1, table2, col2, const2, arithop2

def sort_by_col(data,col_name):
    order = []
    for i in range(0, len(data)):
        order.append(list(data[i]))
        k = order[0].index(col_name)
    for i in range(1, len(order)):
        order[i][k] = int(order[i][k])
    sort = []
    sort.append(order[0])
    temp = (sorted(order[1:], key=(lambda x: [x[k]])))
    for i in range(0, len(temp)):
        sort.append(temp[i])
    res_table = np.asarray(sort)
    return res_table

def join_equal(table1,table2,term, hash_dict, btree_dict,col_dict):
    table1_cols = all_table_dict[table1][0]
    table2_cols = all_table_dict[table2][0]
    res = []
    for i in range(0, len(table1_cols)):
        col_name = table1 + '_' + table1_cols[i]
        res.append(col_name)
    for i in range(0, len(table2_cols)):
        col_name = table2 + '_' + table2_cols[i]
        res.append(col_name)
    print(res)
    mat = []
    mat.append(res)
    terms = term.split('=')
    t1, col1, const1, arithop1, t2, col2, const2, arithop2 = parse_join_terms(terms[0], terms[1])
    data = all_table_dict[t1]
    data2 = all_table_dict[t2]
    col_num1 = find_col_num(data, col1)
    col_num2 = find_col_num(data2, col2)
    if arithop1 == 5:
        tab_col = t1 + '_' + col1
        print(tab_col)
        print(col_dict.values())
        if tab_col in col_dict.keys():
            if col_dict[tab_col] == 'hash':
                print('yes')
                temp_dict = hash_index_dict[tab_col]
                # ??????? how to define
                print(temp_dict.values())
                for i in range(1,len(data2)):
                    print(data2[i][col_num2])
                    if data2[i][col_num2] in temp_dict.values():
                        row = temp_dict[data2[i][col_num2]]
                        print('hash')
                        res.append(list(data[row]))
                print(res)
                datamat = np.array(mat)
                return datamat

        # col1_dict = check_if_index(hash_dict, btree_dict, col_dict, col1, t1)
        # if col1_dict != 'null':
        #     print('col1 has index')
    if arithop2 == 5:
        col2_dict = check_if_index(hash_dict, btree_dict, col_dict, col2, t2)
        if col2_dict != 'null':
            print('col1 has index')
    # if all_table_dict[t1][col_num1].isdigit():
    #     if arithop1 == 5 and arithop2 == 5:
    #         sorted_t1 = sort_by_col(all_table_dict[t1], col1)
    #         sorted_t2 = sort_by_col(all_table_dict[t2], col2)
    #         print(sorted_t1)
    #         i = 1
    #         j = 1
    #         while i < len(sorted_t1) and j < len(sorted_t2):
    #             temp = []
    #             if sorted_t1[i][col_num1] == sorted_t2[j][col_num2]:
    #                 print('flag')
    #                 for m in range(0, len(sorted_t1[i])):
    #                     temp.append(sorted_t1[i][m])
    #                 for n in range(0, len(sorted_t2[j])):
    #                     temp.append(sorted_t2[j][n])
    #                 print(temp)
    #                 mat.append(temp)
    #                 i = i + 1
    #                 j = j + 1
    #             elif sorted_t1[i][col_num1] < sorted_t2[j][col_num2]:
    #                 i = i + 1
    #             elif sorted_t1[i][col_num1] > sorted_t2[j][col_num2]:
    #                 j = j + 1
    #         datamat = np.array(mat)
    #         return datamat
    #     else:
    #         for i in range(1, len(mat)):
    #             left = col_content(mat, arithop1, col1, const1, i)
    #             right = col_content(mat, arithop2, col2, const2, i)
    #             if left <= right:
    #                 #new_res.append(list(mat[i]))
    #                 print('???')


def col_content(data,arithop,col,const,i):
    col_num = find_col_num(data, col)
    if arithop == 1:
        return float(data[i][col_num]) + float(const)
    if arithop == 2:
        return float(data[i][col_num]) - float(const)
    if arithop == 3:
        return float(data[i][col_num]) * float(const)
    if arithop == 4:
        return float(data[i][col_num]) / float(const)

def join_terms(op,terms,mat):
    new_res = []
    new_res.append(list(mat[0]))
    t1, col1, const1, arithop1, t2, col2, const2, arithop2 = parse_join_terms(terms[0], terms[1])
    col_name1 = t1 + '_' + col1
    col_name2 = t2 + '_' + col2
    if op == 1:
        for i in range(1, len(mat)):

            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left <= right:
                new_res.append(list(mat[i]))
    if op == 2:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left >= right:
                new_res.append(list(mat[i]))
    if op == 3:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left != right:
                new_res.append(list(mat[i]))
    if op == 5:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left < right:
                new_res.append(list(mat[i]))
    if op == 6:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left > right:
                new_res.append(list(mat[i]))
    datamat = np.array(new_res)
    print(datamat)
    return datamat






if __name__ == '__main__':
    f2 = open("Tsam.txt", "r")
    lines = f2.readlines()
    print(len(lines))
    filename = 'Sales1.csv'
    filename2 = 'Sales2.csv'
      # 二维列表
    all_table_dict = {}
    hash_index_dict = {}
    Btree_index_dict = {}
    col_dict = {}
    # data = txt_to_matrix(filename)
    # data2 = txt_to_matrix(filename2)
    var = ['Hash', 'R', 'customerid']
    ori_table = var[1]
    index_col = var[2]
    table_col = ori_table + '_' + index_col
    # col = find_col(data, index_col)
    dict = {}
    #index = 0
    # for index in range(0,len(col)):
    #     dict[col[index]] = index
    #     #index = index + 1
    # hash_index_dict[table_col] = dict
    # col_dict[table_col] = 'hash'

    #
    # #moving * 2 !!!!!!!!!
    # item = ['=movavg', 'T2prime', 'R1_qty', '3']
    # col = find_col(data, 'qty')
    # num = list(map(int, col[1:]))
    # #data = np.array(num)
    # res = running_mean(num,int(item[3]))
    # res.insert(0, 'mov_avg')
    # print(col[1:])
    # print(res)
    # data = np.insert(data, len(data[0]), values=res, axis=1)
    # print(data)
    # res2 = moving_sum(num, int(item[3]))
    # print(res2)


    #
    #
    # input = '=join(R,S,R.customerid=S.C)'
    # #input = '=join(R1,S,(R1.qty>S.Q)and(R1.saleid=S.saleid))'
    # order = []
    # for i in range(0, len(data)):
    #     order.append(list(data[i]))
    #     k = order[0].index('customerid')
    # for i in range(1, len(order)):
    #     order[i][k] = int(order[i][k])
    # sort = []
    # sort.append(order[0])
    # temp = (sorted(order[1:], key=(lambda x: [x[k]])))
    # for i in range(0, len(temp)):
    #     sort.append(temp[i])
    # res_table = np.asarray(sort)
    # print(res_table)
    # #
    #
    # if re.search(r'and', input):
    #     temp = re.split(r'and', input)
    #     print('and')
    #     var = re.split(r'[\s\,\(\)]+', temp[0])
    #     var2 = re.split(r'[\s\,\(\)]+', temp[1])
    #     var = var + var2
    #     while '' in var:
    #         var.remove('')
    #     print(var)
    #     table1 = var[1]
    #     table2 = var[2]
    #     for i in range(3,len(var)):
    #         if re.search('=',var[i]):
    #             mat = join_equal(table1, table2, var[i])
    #             print(mat)
    #     for i in range(3,len(var)):
    #         op = relop(var[i], 0)
    #         print(op)
    #         terms = re.split(r'[\=\>\<\!]+', var[i])
    #         if op != 4:
    #             datamat = join_terms(op,terms,mat)
    #             mat = datamat
    #
    #
    # else:
    #     var = re.split(r'[\s\,\(\)]+', input)
    #     while '' in var:
    #         var.remove('')
    #     print(var)
    #     table1 = var[1]
    #     table2 = var[2]
    #     all_table_dict[table1] = data
    #     all_table_dict[table2] = data2
    #     # mat = join_equal(table1,table2,var[3], hash_index_dict, Btree_index_dict,col_dict)
    #     # print(mat)
    #
    #     all_table_dict[table1] = data
    #     all_table_dict[table2] = data2
    #     table1_cols = all_table_dict[table1][0]
    #     table2_cols = all_table_dict[table2][0]
    #     res = []
    #     for i in range(0,len(table1_cols)):
    #         col_name = table1 + '_' + table1_cols[i]
    #         res.append(col_name)
    #     for i in range(0,len(table2_cols)):
    #         col_name = table2 + '_' + table2_cols[i]
    #         res.append(col_name)
    #     print(res)
    #     mat = []
    #     mat.append(res)
    #     terms = var[3].split('=')
    #     t1, col1, const1, arithop1, t2, col2, const2, arithop2 = parse_join_terms(terms[0],terms[1])
    #     sorted_t1 = sort_by_col(all_table_dict[t1], col1)
    #     sorted_t2 = sort_by_col(all_table_dict[t2], col2)
    #     print(sorted_t1)
    #     i = 1
    #     j = 1
    #     col_num1 = find_col_num(data, col1)
    #     col_num2 = find_col_num(data2, col2)
    #     arr1 = find_col(sorted_t1, col1)
    #     arr2 = find_col(sorted_t2, col2)
    #     print(len(arr1))
    #     print(len(arr2))
    #     count = 0
    #     res = []
    #     for l in arr1:
    #         if l in arr2:
    #             print(l)
    #             res.append(l)
    #             count=count+1
    #     print(count)
    #     print(res)
    #     #print(len(set1 & set2))
    #
    #     while i < len(sorted_t1) and j < len(sorted_t2):
    #         # print(i)
    #         # print(j)
    #         temp = []
    #         if sorted_t1[i][col_num1] == sorted_t2[j][col_num2]:
    #             #print('flag')
    #             for m in range(0, len(sorted_t1[i])):
    #                 temp.append(sorted_t1[i][m])
    #             for n in range(0, len(sorted_t2[j])):
    #                 temp.append(sorted_t2[j][n])
    #             #print(temp)
    #             mat.append(temp)
    #             i = i + 1
    #             j = j + 1
    #         elif sorted_t1[i][col_num1] < sorted_t2[j][col_num2]:
    #             i = i + 1
    #         elif sorted_t1[i][col_num1] > sorted_t2[j][col_num2]:
    #             j = j + 1
    #     datamat = np.array(mat)
    #     print(len(datamat))
    #     # if op is <=

    # input = '=select(R,(time>50)or(qty<30))'
    #
    # if re.search(r'or', input):
    #     temp = re.split(r'or', input)
    #     print(temp)
    #     var = re.split(r'[\s\,\(\)]+', temp[0])
    #     var2 = re.split(r'[\s\,\(\)]+', temp[1])
    #     var = var+var2
    #     res = []
    #     res.append(list(data[0]))
    #     while '' in var:
    #         var.remove('')
    #     print(var)
    #     for i in range(2,len(var)):
    #         print(i)
    #         datamat = parse_condition(var[1], var[2], res, data)
    #         print(datamat)
    # elif re.search(r'and', input):
    #     temp = re.split(r'and', input)
    #     print('and')
    #     var = re.split(r'[\s\,\(\)]+', temp[0])
    #     var2 = re.split(r'[\s\,\(\)]+', temp[1])
    #     var = var+var2
    #     while '' in var:
    #         var.remove('')
    #     print(var)
    #     for i in range(2,len(var)):
    #         res = []
    #         res.append(list(data[0]))
    #         print(i)
    #         datamat = parse_condition(var[1], var[2], res, data)
    #         data = datamat
    #         print(data)
    # else:
    #     var = re.split(r'[\s\,\(\)]+', input)
    #     while '' in var:
    #         var.remove('')
    #     print(var)
    #     res = []
    #     res.append(list(data[0]))
    #     datamat = parse_condition(var[1], var[2],res, data)
    #     print(datamat)
    # print(type(datamat))

    # save file
    # fileObject = open('sampleList.txt', 'w')
    # for i in range(0,len(datamat)):
    #     fileObject.write(str(datamat[i]))
    #     fileObject.write('\n')
    #
    # fileObject.close()
    #np.savetxt('selector.txt', datamat)
        # #identify condition
        # terms = re.split(r'[\=\>\<\!]+', var[2])
        # loc, col, const, arithop, constant = parse_terms(terms)
        # print(loc)
        # print(col)
        # print(const)
        # print(constant)
        # print(arithop)
        # print(terms)
        # res = []
        # res.append(list(data[0]))
        # # identify op
        # op = relop(var[2],loc)
        # print(op)
        # # if op is <=
        # if op == 1:
        #     col_num = find_col_num(data, col)
        #     for i in range(1, len(data)):
        #         if arithop == 1:
        #             if int(data[i][col_num]) + int(const) <= int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 2:
        #             if int(data[i][col_num]) - int(const) <= int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 3:
        #             if int(data[i][col_num]) * int(const) <= int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 4:
        #             if int(data[i][col_num]) / int(const) <= int(constant):
        #                 res.append(list(data[i]))
        # # if op is >=
        # if op == 2:
        #     col_num = find_col_num(data, terms[0])
        #     for i in range(1, len(data)):
        #         if arithop == 1:
        #             if int(data[i][col_num]) + int(const) >= int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 2:
        #             if int(data[i][col_num]) - int(const) >= int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 3:
        #             if int(data[i][col_num]) * int(const) >= int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 4:
        #             if int(data[i][col_num]) / int(const) >= int(constant):
        #                 res.append(list(data[i]))
        # # if op is !=
        # if op == 3:
        #     col_num = find_col_num(data, terms[0])
        #     for i in range(1, len(data)):
        #         if arithop == 1:
        #             if int(data[i][col_num]) + int(const) != int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 2:
        #             if int(data[i][col_num]) - int(const) != int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 3:
        #             if int(data[i][col_num]) * int(const) != int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 4:
        #             if int(data[i][col_num]) / int(const) != int(constant):
        #                 res.append(list(data[i]))
        # # if op is =
        # if op == 4:
        #     # index
        #     tab_col = var[1] + '_' + col
        #     print(tab_col)
        #     if tab_col in col_dict.keys():
        #         if col_dict[tab_col] == 'hash':
        #             temp_dict = hash_index_dict[tab_col]
        #             # ??????? how to define
        #             row = temp_dict[constant]
        #             print(row)
        #             res.append(list(data[row]))
        #             print(res)
        #     # no index
        #     else:
        #         # identify column
        #         col_num = find_col_num(data, terms[0])
        #         for i in range(0, len(data)):
        #             if arithop == 1:
        #                 if int(data[i][col_num]) + int(const) == int(constant):
        #                     res.append(list(data[i]))
        #             if arithop == 2:
        #                 if int(data[i][col_num]) - int(const) == int(constant):
        #                     res.append(list(data[i]))
        #             if arithop == 3:
        #                 if int(data[i][col_num]) * int(const) == int(constant):
        #                     res.append(list(data[i]))
        #             if arithop == 4:
        #                 if int(data[i][col_num]) / int(const) == int(constant):
        #                     res.append(list(data[i]))
        # # if op is <
        # if op == 5:
        #     col_num = find_col_num(data, terms[0])
        #     for i in range(1, len(data)):
        #         if arithop == 1:
        #             if int(data[i][col_num]) + int(const) < int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 2:
        #             if int(data[i][col_num]) - int(const) < int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 3:
        #             if int(data[i][col_num]) * int(const) < int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 4:
        #             if int(data[i][col_num]) / int(const) < int(constant):
        #                 res.append(list(data[i]))
        # # if op is >
        # if op == 6:
        #     col_num = find_col_num(data, terms[0])
        #     for i in range(1, len(data)):
        #         if arithop == 1:
        #             if int(data[i][col_num]) + int(const) > int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 2:
        #             if int(data[i][col_num]) - int(const) > int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 3:
        #             if int(data[i][col_num]) * int(const) > int(constant):
        #                 res.append(list(data[i]))
        #         if arithop == 4:
        #             if int(data[i][col_num]) / int(const) > int(constant):
        #                 res.append(list(data[i]))
        # datamat = np.array(res)
        # print(datamat)

       # if re.search(r'[\+\-\*\/]+', var[2]):
       #      loc, col, const, arithop, constant = parse_terms(terms)
       #      print(loc)
       #      print(col)
       #      print(const)
       #      print(constant)
       #      print(arithop)
       #      print(terms)
       #      res = []
       #      res.append(list(data[0]))
       #      # identify op
       #      op = relop(var[2])
       #      print(op)
       #      # if op is <=
       #      if op == 1:
       #          col_num = find_col_num(data, col)
       #          for i in range(1, len(data)):
       #              if arithop == 1:
       #                  if int(data[i][col_num]) + int(const) <= int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 2:
       #                  if int(data[i][col_num]) - int(const) <= int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 3:
       #                  if int(data[i][col_num]) * int(const) <= int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 4:
       #                  if int(data[i][col_num]) / int(const) <= int(constant):
       #                      res.append(list(data[i]))
       #      # if op is >=
       #      if op == 2:
       #          col_num = find_col_num(data, terms[0])
       #          for i in range(1, len(data)):
       #              if arithop == 1:
       #                  if int(data[i][col_num]) + int(const) >= int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 2:
       #                  if int(data[i][col_num]) - int(const) >= int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 3:
       #                  if int(data[i][col_num]) * int(const) >= int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 4:
       #                  if int(data[i][col_num]) / int(const) >= int(constant):
       #                      res.append(list(data[i]))
       #      # if op is !=
       #      if op == 3:
       #          col_num = find_col_num(data, terms[0])
       #          for i in range(1, len(data)):
       #              if arithop == 1:
       #                  if int(data[i][col_num]) + int(const) != int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 2:
       #                  if int(data[i][col_num]) - int(const) != int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 3:
       #                  if int(data[i][col_num]) * int(const) != int(constant):
       #                      res.append(list(data[i]))
       #              if arithop == 4:
       #                  if int(data[i][col_num]) / int(const) != int(constant):
       #                      res.append(list(data[i]))
       #      # if op is =
       #      if op == 4:
       #          # index
       #          tab_col = var[1] + '_' + terms[0]
       #          print(tab_col)
       #          if tab_col in col_dict.keys():
       #              if col_dict[tab_col] == 'hash':
       #                  temp_dict = hash_index_dict[tab_col]
       #                  row = temp_dict[terms[1]]
       #                  print(row)
       #                  res.append(list(data[row]))
       #                  print(res)
       #          # no index
       #          else:
       #              # identify column
       #              col_num = find_col_num(data, terms[0])
       #              for i in range(0, len(data)):
       #                  if data[i][col_num] == terms[1]:
       #                      res.append(list(data[i]))
       #      if op == 5:
       #          col_num = find_col_num(data, terms[0])
       #          for i in range(1, len(data)):
       #              if int(data[i][col_num]) < int(terms[1]):
       #                  res.append(list(data[i]))
       #      if op == 6:
       #          col_num = find_col_num(data, terms[0])
       #          for i in range(1, len(data)):
       #              if int(data[i][col_num]) > int(terms[1]):
       #                  res.append(list(data[i]))
       #      datamat = np.array(res)
       #      print(datamat)
        # else:
        #     datamat = parse_condition(var[2])
        #     print(datamat)





    # item = ['=project', 'R1', 'saleid', 'qty', 'pricerange']
    # record = []
    # for i in range(2,len(item)):
    #     col = find_col(data,item[i])
    #     print(col)
    #     record.append(col)
    # datamat = np.array(record)
    # mat = datamat.T
    # print(mat)

    # item = ['=avg', 'R', 'qty']
    # col = find_col(data, item[2])
    # print(col)
    # name = 'avg('+item[2]+')'
    # record = []
    # record.append(name)
    # #avg = np.mean(col)
    # num = arr = list(map(int,col[1:]))
    # print(num)
    # #print(avg)
    # record.append(mean(num))
    # print(record)
    # datamat = np.array(record)
    # print(datamat)
    # mat = datamat.reshape(datamat.shape[0],1)
    # print(mat)
    #
    # item = ['=sort', 'T1', 'saleid', 'qty']
    # print(list(data)[1:][:-1])
    # #res = sort_by_col(data[1:], 1)
    # num = list(map(int,data[1:][:-1]))
    # print(num)



    # all_index_dict = {}
    # btree_dict = {}
    # var = ['Hash', 'R', 'itemid']
    # ori_table = var[1]
    # index_col = var[2]
    # col = find_col(data, 'qty')
    # dict = {}
    # #index = 0
    # for index in range(0,len(col)):
    #     dict[col[index]] = index
    #     #index = index + 1
    # all_index_dict['r'] = dict
    # t = OIBTree()
    # t.update(dict)
    # # print(dict)
    # # print(all_index_dict)
    # btree_dict[index_col] = t
    # print(dict)
    # print(btree_dict)
    #
    # data2 = data[1:]
    # print(len(data2))
    # res = np.concatenate((data,data2),axis=0)
    # print(len(res))






