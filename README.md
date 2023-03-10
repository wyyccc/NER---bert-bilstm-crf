# NER
for ner task

model: bert-bilstm-crf

env: pip install torchcrf==0.4.0

## stage 1

max_len of sentence = 500
lstm_hidden_size = 128

For this stage, optimization mainly focuses on training and modeling.

| version | model | train | other | f1 | comment |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------:|
| v1 | bert-base-chinese | plain | baseline | 0.7848 | baseline |
| v2 | bert-base-chinese | linear lr & Lookahead & FGM | 8 epoch | 0.7984 | **linear lr & Lookahead & FGM improves f1** |
| v3 | bert-base-chinese + BERT 2-layer (add) | linear lr & Lookahead & FGM | 8 epoch | 0.8016 | **BERT 2-layer (addition) improves f1** |
| v4 | ernie3.0 + BERT 2-layer (add) | linear lr & Lookahead & FGM | 8 epoch | 0.8074 | **ernie3.0 is better** |

## stage 2

********data processing improved & evaluation debuged********

evaluation function had bugs, there was an error in judging continuous entities. It was solved in v5.

For this stage, optimization mainly focuses on data input and training.

| version | model | train | other | f1 | comment |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------:|
| v5 | ernie3.0 + BERT 2-layer (add) | linear lr & Lookahead & FGM | 16 epoch | 0.8504 | baseline for this stage |
| v6 | ernie3.0 + BERT 2-layer (add) | linear lr & Lookahead & FGM | 16 epoch & BIOE | 0.8234 | BIOE doesn't work |
| v7 | ernie3.0 + BERT 2-layer (add) | hierarchical and linear lr & Lookahead & FGM | 8 epoch | 0.8506 | **hierarchical and linear lr accelerates convergence** |

## stage 3

********max_len of sentence changed to 450********

idea(prompt): Splice types: (type1 [SEP] type2 [SEP] ... typen [SEP]) to the beginning of sentences, label set: "P".

For this stage, optimization mainly focues on modeling.

| version | model | train | other | f1 | comment |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------:|
| v8 | ernie3.0 + BERT 2-layer (add) | hierarchical and linear lr & Lookahead & FGM | 8 epoch | 0.8500 | baseline for this stage |
| v9 | ernie3.0 + BERT 2-layer (add) | linear lr & Lookahead & FGM | 8 epoch & Prompt | 0.8508 | **prompt improves f1 slightly** (baseline for this stage) |
| v10 | ernie3.0 + BERT 4-layer (add) | linear lr & Lookahead & FGM | 8 epoch & Prompt | 0.8484 | add more layers not helps |
| v11 | ernie3.0 + BERT 2-layer (concat) | linear lr & Lookahead & FGM | 8 epoch & Prompt | 0.8527 | **concat is better than add** |
| v12 | ernie3.0 + BERT 4-layer (concat) | linear lr & Lookahead & FGM | 8 epoch & Prompt | 0.8525 | more layers need more epoch |
| v13 | ernie3.0 + BERT 4-layer (concat) | linear lr & Lookahead & FGM | 12 epoch & Prompt | 0.8530 | more layers may not help |
| v14 | ernie3.0 + BERT 4-layer (concat) + self-attention | linear lr & Lookahead & FGM | 12 epoch & Prompt | 0.8497 | additional self-attention doesn't help |
| v15 | ernie3.0 + BERT 2-layer (concat) + lstm(64/256) | linear lr & Lookahead & FGM | 8 epoch & Prompt | 0.8503/0.8508 | different lstm hidden_size doesn't help |

## stage 4

For this stage, backbone changes to chinese-roberta-wwm-ext-large.

| version | model | train | other | f1 | comment |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------:|
| v16 | roberta-large + BERT 2-layer (concat) | hierarchical and linear lr & Lookahead & FGM | 8 epoch & Prompt |  | baseline for this stage |



