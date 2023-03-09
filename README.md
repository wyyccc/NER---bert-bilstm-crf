# NER
for ner task

model: bert-bilstm-crf

env: pip install torchcrf==0.4.0

********baseline********

max_len of sentence: 500

| version | model | train | other | f1 |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|
| v1 | bert-base-chinese | plain | baseline | 0.7848 |
| v2 | bert-base-chinese | linear lr & Lookahead & FGM | 8 epoch | 0.7984 |
| v3 | bert-base-chinese + BERT 2-layer (addition) | linear lr & Lookahead & FGM | 8 epoch | 0.8016 |
| v4 | ernie3.0 + BERT 2-layer (addition) | linear lr & Lookahead & FGM | 8 epoch | 0.8074 |

********data processing improved & evaluation debuged********

| version | model | train | other | f1 |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|
| v5 | ernie3.0 + BERT 2-layer (addition) | linear lr & Lookahead & FGM | 16 epoch | 0.8504 |
| v6 | ernie3.0 + BERT 2-layer (addition) | linear lr & Lookahead & FGM | 16 epoch & BIOE | 0.8234 |
| v7 | ernie3.0 + BERT 2-layer (addition) | hierarchical and linear lr & Lookahead & FGM | 16 epoch | 0.8506 |

********max_len of sentence changed to 450********

idea(prompt): Splice types: (type1 [SEP] type2 [SEP] ... typen [SEP]) to the beginning of sentences, label set: "P".

| version | model | train | other | f1 |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|
| v8 | ernie3.0 + BERT 2-layer (addition) | hierarchical and linear lr & Lookahead & FGM | 16 epoch | 0.8506 |
| v9 | ernie3.0 + BERT 2-layer (addition) | linear lr & Lookahead & FGM | 16 epoch & Prompt | 0.8234 |
