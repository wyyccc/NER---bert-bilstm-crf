import json
import pandas as pd
import numpy as np

stop_syn = ['，', '；', ',', ';', '、','：',':','）',')']

def data_process_BIO(data):
    x_list = [i for i in data['text']]
    y_list = ['O'] * len(x_list)
    for tag in data['annoResult']:
        tag = json.loads(tag)
        label = tag['labelId']
        if label == '10':
            continue
        start = int(tag['start'])
        end = int(tag['end'])
        y_list[start] = 'B-' + label
        for i in range(start+1, end):
            y_list[i] = 'I-' + label
    return x_list, y_list

def data_process_BIOE(data):
    x_list = [i for i in data['text']]
    y_list = ['O'] * len(x_list)
    for tag in data['annoResult']:
        tag = json.loads(tag)
        label = tag['labelId']
        if label == '10':
            continue
        start = int(tag['start'])
        end = int(tag['end'])
        y_list[start] = 'B-' + label
        y_list[end-1] = 'E-' + label
        for i in range(start+1, end-1):
            y_list[i] = 'I-' + label
    return x_list, y_list

def data_prepare(data, type = 'BIO'):
    x_list = []
    y_list = []
    for text in data['textList']:
        if type == 'BIO':
            x_list.append(data_process_BIO(text)[0])
            y_list.append(data_process_BIO(text)[1])
        elif type == 'BIOE':
            x_list.append(data_process_BIOE(text)[0])
            y_list.append(data_process_BIOE(text)[1])
    return x_list, y_list

def seq_cut(data, is_train = True):
    x_list = []
    y_list = []
    error = 0
    if is_train:
        for i in range(len(data[0])):
            if len(data[0][i]) < 500:
                x_list.append(data[0][i])
                y_list.append(data[1][i])
            else:
                temp_x = data[0][i]
                temp_y = data[1][i]
                while len(temp_x)>=500:
                    for j in range(499,0,-1):
                        if temp_x[j]=='。':
                            break
                    if j == 1:
                        for j in range(499,0,-1):
                            if temp_x[j] in stop_syn:
                                break
                    if j == 1:
                        error += 1
                        break
                    else:
                        x_list.append(temp_x[:j+1])
                        y_list.append(temp_y[:j+1])
                    temp_x = temp_x[j+1:]
                    temp_y = temp_y[j+1:]
                if len(temp_x) < 500:
                    x_list.append(temp_x)
                    y_list.append(temp_y)
        print('train seq cut error: ', error)
        return x_list, y_list
    else:
        for i in range(len(data)):
            if len(data[i]) < 500:
                x_list.append(data[i])
            else:
                temp_x = data[i]
                while len(temp_x)>=500:
                    for j in range(499,0,-1):
                        if temp_x[j]=='。':
                            break
                    if j == 1:
                        for j in range(499,0,-1):
                            if temp_x[j] in stop_syn:
                                break
                    if j == 1:
                        error += 1
                        break
                    else:
                        x_list.append(temp_x[:j+1])
                    temp_x = temp_x[j+1:]
                if len(temp_x) < 500:
                    x_list.append(temp_x)
        print('test seq cut error: ', error)
        return x_list

def data_train(type = 'BIO'):
    data1 = json.load(open('/mnt/disk2/wyc/ner/data/0-500.json', 'r', encoding = 'utf-8'))
    x_list1, y_list1 = data_prepare(data1, type)
    data2 = json.load(open('/mnt/disk2/wyc/ner/data/500-1000.json', 'r', encoding = 'utf-8'))
    x_list2, y_list2 = data_prepare(data2, type)
    data3 = json.load(open('/mnt/disk2/wyc/ner/data/1000-1500.json', 'r', encoding = 'utf-8'))
    x_list3, y_list3 = data_prepare(data3, type)
    data4 = json.load(open('/mnt/disk2/wyc/ner/data/1500-2000.json', 'r', encoding = 'utf-8'))
    x_list4, y_list4 = data_prepare(data4, type)
    x_list = x_list1 + x_list2 + x_list3 + x_list4
    y_list = y_list1 + y_list2 + y_list3 + y_list4
    data = seq_cut([x_list, y_list])
    return data

def data_eval(type = 'BIO'):
    data1 = json.load(open('/mnt/disk2/wyc/ner/data/2000-2500.json', 'r', encoding = 'utf-8'))
    x_list1, y_list1 = data_prepare(data1, type)
    data2 = json.load(open('/mnt/disk2/wyc/ner/data/2500-3000.json', 'r', encoding = 'utf-8'))
    x_list2, y_list2 = data_prepare(data2, type)
    x_list = x_list1 + x_list2
    y_list = y_list1 + y_list2
    data = seq_cut([x_list, y_list])
    return data

if __name__ == '__main__':
    '''
    data_train()返回[[texts],[tags]]
    data_eval()返回[[texts],[tags]]
    '''
    train = data_train(type = 'BIOE')
    eval = data_eval()
    print(train[1][1])
