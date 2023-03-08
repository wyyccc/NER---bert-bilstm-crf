# NER
for ner task

model: bert-bilstm-crf

env: pip install torchcrf==0.4.0
    
| version | model | train | other | f1 |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|
| v1 | bert-base-chinese | plain | baseline | 0.7848 |
| v2 | bert-base-chinese | linear lr & Lookahead & FGM | 8 epoch | 0.7984 |
| v3 | bert-base-chinese + BERT 2-layer | linear lr & Lookahead & FGM | 8 epoch | 0.8016 |
| v4 | ernie3.0 + BERT 2-layer | linear lr & Lookahead & FGM | 8 epoch | 0.8074 |

********data processing improved & evaluation debuged********


| version | model | train | other | f1 |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|
| v5 | ernie3.0 + BERT 2-layer | linear lr & Lookahead & FGM | 16 epoch | 0.8504 |
| v6 | ernie3.0 + BERT 2-layer | linear lr & Lookahead & FGM | 16 epoch & BIOE | 0.8234 |
| v7 | ernie3.0 + BERT 2-layer | hierarchical and linear lr & Lookahead & FGM | 16 epoch |  |
