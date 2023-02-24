from helper.data_preprocess import data_train, data_test
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, BertTokenizer
from torchcrf import CRF
from helper.lookahead import Lookahead
from helper.fgm import FGM
from helper.eval import evaluation_entity
from seqeval.metrics import f1_score

model_path = '/mnt/disk2/wyc/pretrained-models/bert-base-chinese-ner'
hidden_dropout_prob = 0.1
hidden_size = 768

MODEL_PATH = '/mnt/disk2/wyc/ner/model/bert-bilstm-crf'
result_path = '/mnt/disk2/wyc/ner/result/bert-bilstm-crf.csv'

seed = 41
MAX_LEN = 512
BATCH_SIZE = 32
EPOCH = 8
lr = 4e-5
min_lr = 4e-6
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# tag2index
tag2index = {
    "O": 0,
    "B-6": 1, "I-6": 2,
    "B-7": 3, "I-7": 4,
    "B-8": 5, "I-8": 6,
    "B-9": 7, "I-9": 8,
    "B-11": 9, "I-11": 10,
    "B-12": 11, "I-12": 12,
    "B-13": 13, "I-13": 14,
    "B-14": 15, "I-14": 16,
}
index2tag = {v: k for k, v in tag2index.items()}

def seed_everything(seed=seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def data_preprocessing(data, is_train):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    token_texts = []
    if is_train:
        texts = data[0]
    else:
        texts = data
    for text in tqdm(texts):
        tokenized = tokenizer.encode_plus(text=text,
                                          max_length=MAX_LEN,
                                          return_token_type_ids=True,
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                          padding='max_length',
                                          truncation=True)
        token_texts.append(tokenized)
    tags = None
    if is_train:
        tags = []
        for tag in tqdm(data[1]):
            index_list = [0] + [tag2index[t] for t in tag] + [0]
            if len(index_list) < MAX_LEN:  # 填充
                pad_length = MAX_LEN - len(index_list)
                index_list += [tag2index['O']] * pad_length
            if len(index_list) > MAX_LEN:  # 裁剪
                index_list = index_list[:MAX_LEN-1] + [0]
            tags.append(index_list)
        tags = torch.LongTensor(tags)
    return token_texts, tags

class NerDataset(Dataset):
    def __init__(self, token_texts, tags):
        super(NerDataset, self).__init__()
        self.token_texts = token_texts
        self.tags = tags

    def __getitem__(self, index):
        return {
            "token_texts": self.token_texts[index],
            "tags": self.tags[index] if self.tags is not None else None,
        }

    def __len__(self):
        return len(self.token_texts)


class NerDatasetTest(Dataset):
    def __init__(self, token_texts):
        super(NerDatasetTest, self).__init__()
        self.token_texts = token_texts

    def __getitem__(self, index):
        return {
            "token_texts": self.token_texts[index],
            "tags": 0
        }

    def __len__(self):
        return len(self.token_texts)


class Bert_BiLSTM_CRF(nn.Module):
    '''
    bert last 4-layers sum
    bilstm
    crf
    '''
    def __init__(self, tag2index):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tagset_size = len(tag2index)
        self.bert = AutoModel.from_pretrained(model_path, output_hidden_states = True)
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.1)
        self.dense = nn.Linear(in_features=256, out_features=self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size, batch_first=True)
        self.hidden = None
    
    def forward(self, token_texts, tags):
        texts, token_type_ids, masks = token_texts.values()
        texts = texts.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        masks = masks.squeeze(1)
        bert_out_all = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)
        bert_out = bert_out_all[0]
        bert_out += bert_out_all[2][11] + bert_out_all[2][10] + bert_out_all[2][9]
        bert_out = bert_out.permute(1, 0, 2)
        device = bert_out.device
        self.hidden = (torch.randn(2, bert_out.size(0), 128).to(device),
                       torch.randn(2, bert_out.size(0), 128).to(device))
        out, self.hidden = self.lstm(bert_out, self.hidden)
        lstm_feats = self.dense(out)
        lstm_feats = lstm_feats.permute(1, 0, 2)
        masks = masks.clone().detach().bool()
        # 计算损失值和预测值
        if tags is not None:
            predictions = self.crf.decode(lstm_feats, mask=masks)
            loss = -self.crf(lstm_feats, tags, masks)
            return loss, predictions
        else:
            predictions = self.crf.decode(lstm_feats, mask=masks)
            return predictions

def get_f1_score(tags, predictions):
    tags = tags.to('cpu').data.numpy().tolist()
    temp_tags = []
    final_tags = []
    for index in range(BATCH_SIZE):
        predictions[index].pop()
        length = len(predictions[index])
        temp_tags.append(tags[index][1:length])
        predictions[index].pop(0)
        temp_tags[index] = [index2tag[x] for x in temp_tags[index]]
        predictions[index] = [index2tag[int(x)] for x in predictions[index]]
        final_tags.append(temp_tags[index])
    f1 = f1_score(final_tags, predictions, average='micro')
    return f1

def train_epoch(train_dataloader, model, optimizer, epoch, start):
    fgm = FGM(model)
    for i, batch_data in enumerate(train_dataloader):
        token_texts = dict()
        token_texts['input_ids'] = batch_data['token_texts']['input_ids'].to(DEVICE)
        token_texts['token_type_ids'] = batch_data['token_texts']['token_type_ids'].to(DEVICE)
        token_texts['attention_mask'] = batch_data['token_texts']['attention_mask'].to(DEVICE)
        tags = batch_data['tags'].to(DEVICE)
        loss, predictions = model(token_texts, tags)
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        # FGM
        fgm.attack()
        loss_adv, _ = model(token_texts, tags)
        loss_adv.backward(torch.ones_like(loss_adv))
        fgm.restore() 
        optimizer.step()
        if i % 50 == 0:
            micro_f1 = get_f1_score(tags, predictions)
            now = time.time()
            s = round(now - start, 1)
            print(f'Epoch:{epoch} | i:{i} | loss:{loss} | Micro_F1:{micro_f1} | Elapse:{s}s')

def predict(test_dataloader, model):
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for i, batch_data in enumerate(test_dataloader):
            token_texts = dict()
            token_texts['input_ids'] = batch_data['token_texts']['input_ids'].to(DEVICE)
            token_texts['token_type_ids'] = batch_data['token_texts']['token_type_ids'].to(DEVICE)
            token_texts['attention_mask'] = batch_data['token_texts']['attention_mask'].to(DEVICE)
            predictions = model(token_texts, None)
            predictions_list.extend(predictions)
    print(len(predictions_list))
    entity_tag_list = []
    index2tag = {v: k for k, v in tag2index.items()}  # 反转字典
    for i in range(len(predictions_list)):
        predictions = predictions_list[i]
        predictions.pop()
        predictions.pop(0)
        text_entity_tag = []
        for pred in predictions:
            text_entity_tag.append(index2tag[pred])
        entity_tag_list.append(" ".join(text_entity_tag))
    result_df = pd.DataFrame(data=entity_tag_list, columns=['result'])
    return result_df

def train():
    train_dataset = data_train()
    token_texts, tags = data_preprocessing(train_dataset, is_train=True)
    train_dataset = NerDataset(token_texts, tags)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = Bert_BiLSTM_CRF(tag2index=tag2index).to(DEVICE)
    print(f"GPU_NAME:{torch.cuda.get_device_name()} | Memory_Allocated:{torch.cuda.memory_allocated()}")
    # eval dataset
    test_data, label = data_test()
    test_token_texts, _ = data_preprocessing(test_data, is_train=False)
    test_dataset = NerDatasetTest(test_token_texts)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # train loop
    f1 = 0
    for i in range(EPOCH):
        model.train()
        start = time.time()
        if i < EPOCH/2:
            lr_temp = min_lr + (lr-min_lr)*i/EPOCH*2
        else:
            lr_temp = min_lr + (lr-min_lr)*(EPOCH-i)/EPOCH*2
        optimizer = optim.AdamW(model.parameters(), lr = lr_temp)
        print('lr:', lr_temp)
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
        train_epoch(train_dataloader, model, optimizer, i, start)
        # eval
        result_df = predict(test_dataloader, model)
        result = list(result_df['result'])
        for i in tqdm(range(len(result))):
            result[i] = result[i].split(' ')
        f1_epoch = evaluation_entity(result, label)
        # save model
        if f1_epoch > f1:
            torch.save(model.state_dict(), MODEL_PATH)
            f1 = f1_epoch
    
def test():
    test_dataset, _ = data_test()
    token_texts, _ = data_preprocessing(test_dataset, is_train=False)
    test_dataset = NerDatasetTest(token_texts)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = Bert_BiLSTM_CRF(tag2index).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    # 模型预测
    result_df = predict(test_dataloader, model)
    result_df.to_csv(result_path)


if __name__ == '__main__':
    train()
    test()