import pandas as pd
from tqdm import tqdm
from data_preprocess_eval import data_eval
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
    for i in tqdm(range(len(pred))):
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
    return 0


if __name__ == '__main__':
    csv_path = '/mnt/disk2/wyc/ner/result/bert-bilstm-crf-v3.csv'
    pred = read_data(csv_path)
    data, true = data_eval()
    evaluation(pred, true)
    evaluation_entity(pred, true)
    
    '''
    v1: bert-base-chinese / baseline / epoch 10 / lr 1e-5 -- 1e-6
    precison: 0.7853338197738052    recall 0.7843227749708511    f1: 0.7848279717560128
           O        0.98      0.98      0.98     1063275
          人名       0.98      0.98      0.98      3074
          会议       0.87      0.83      0.85      5447
          地名       0.94      0.87      0.90     11864
        政策倾向      0.74      0.76      0.75     36115
        政策词汇      0.83      0.65      0.73      8303
          时间       0.97      0.94      0.96     20975
        核心观点      0.97      0.98      0.97     25122
        组织机构      0.93      0.92      0.93     88824
        accuracy                          0.97   1262999
       macro avg      0.91      0.88      0.89   1262999
    weighted avg      0.97      0.97      0.97   1262999
        
    v2: bert-base-chinese / linear lr & Lookahead & FGM / epoch 5 / lr 2e-6 -- 2e-5 -- 2e-6
    precison: 0.8042144177449169    recall 0.7926301982122037    f1: 0.7983802894437444
           O        0.98      0.98      0.98     1063275
          人名       0.98      0.98      0.98      3074
          会议       0.86      0.90      0.88      5447
          地名       0.92      0.90      0.91     11864
        政策倾向      0.76      0.74      0.75     36115
        政策词汇      0.84      0.66      0.74      8303
          时间       0.98      0.93      0.95     20975
        核心观点      0.97      0.99      0.98     25122
        组织机构      0.92      0.95      0.94     88824
        accuracy                         0.97    1262999
       macro avg      0.91      0.89      0.90   1262999
    weighted avg      0.97      0.97      0.97   1262999
    
    v3: bert-base-chinese / linear lr & Lookahead & FGM & BERT 4-layer / epoch 8 / lr 2e-6 -- 2e-5 -- 2e-6
    precison: 0.7952060856876301    recall 0.8074718227749709    f1: 0.8012920176929288
           O         0.98      0.98      0.98   1063275
          人名       0.98      0.99      0.98      3074
          会议       0.86      0.85      0.86      5447
          地名       0.93      0.89      0.91     11864
        政策倾向      0.75      0.75      0.75     36115
        政策词汇      0.82      0.72      0.76      8303
          时间       0.97      0.95      0.96     20975
        核心观点      0.96      0.99      0.98     25122
        组织机构      0.93      0.94      0.94     88824

        accuracy                           0.97   1262999
       macro avg       0.91      0.90      0.90   1262999
      weighted avg     0.97      0.97      0.97   1262999
    '''