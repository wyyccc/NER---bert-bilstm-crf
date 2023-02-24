import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

tag2index = {
    "O": "O",
    "B-6": "人名", "I-6": "人名",
    "B-7": "地名", "I-7": "地名",
    "B-8": "时间", "I-8": "时间",
    "B-9": "会议", "I-9": "会议",
    "B-11": "核心观点", "I-11": "核心观点",
    "B-12": "组织机构", "I-12": "组织机构",
    "B-13": "政策词汇", "I-13": "政策词汇",
    "B-14": "政策倾向", "I-14": "政策倾向",
}

def read_data(csv_path):
    result = list(pd.read_csv(csv_path)['result'])
    for i in tqdm(range(len(result))):
        result[i] = result[i].split(' ')
    return result

def evaluation(pred, true):
    pred = sum(pred, [])
    true = sum(true, [])
    for i in tqdm(range(len(pred))):
        pred[i] = tag2index[pred[i]]
        true[i] = tag2index[true[i]]
    print(classification_report(true, pred))
    print('acc:', accuracy_score(pred, true))
    print('precision:', metrics.precision_score(true, pred, average="micro"))
    print('recall:', metrics.precision_score(true, pred, average="micro"))
    print('f1:', metrics.f1_score(true, pred, average="micro"))
    return 0

def evaluation_entity(pred, true):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(pred)):
        temp_pred = pred[i]
        temp_true = true[i]
        tp = 0
        fp = 0
        gt = 0
        for j in range(len(temp_true)):
            if temp_true[j][0] == 'B':
                gt += 1
        for j in range(len(temp_pred)):
            if temp_pred[j][0] == 'B':
                temp = True
                for k in range(j, len(temp_pred)):
                    if temp_pred[k] == 'O':
                        if temp_true[k] != 'O':
                            temp = False
                        break
                    if temp_pred[k] != temp_true[k]:
                        temp = False
                        break
                if temp:
                    tp += 1
                else:
                    fp += 1
        fn = gt - tp
        TP += tp
        FP += fp
        FN += fn
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    print('precison:', precision)
    print('recall', recall)
    print('f1:', f1)
    return f1