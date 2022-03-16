# Event-Centric Opinion Mining
- An An implementation for EcO v1: Towards Event-Centric Opinion Mining.
- Please contact @Ruoxi Xu (ruoxi2021@iscas.ac.cn) for questions and suggestions.

## Requirements
General
- Python (verified on 3.8)
- CUDA (verified on 11.1)

Python Packages
- see requirements.txt

```python
conda create -n ecom python=3.8
conda activate ecom

pip install -r requirements.txt
```

## Quick Start
### Data Format
Data folder contains two folders: ECO-EN and ECO-ZH.

Before training models, you should first download [data](http://123.57.148.143/EcO_bank.zip) and unzip them as follows. 
```
data
├── ECOB-ZH  # Chinese dataset.
├── ── train.json
├── ── dev.json
├── ── test.json
├──   ECOB-EN  # English dataset.
├── ── train.json
├── ── dev.json
└── ── test.json
```

The data format is as follows:
```
{
    "Descriptor": {
        "text": "22家电商被约谈",
        "candidate_arguments": [
            "电商",
            "22家电商",
            "22家电商被约谈"
        ]
    },
    "Doc": {
        "id": 2109,
        "title": "双11将至，这22家电商被约谈！",
        "content": [
            [
                "11月1日，南都记者从上海市市场监管局获悉，日前，该局联合多部门召集途虎养车、美团点评、国美在线、小红书、健一网、药房网、洋码头等22家电商企业召开“规范‘双十一’网络集中促销活动”行政约谈会。",
                "O",
                "O"
            ],
            ...
            [
                "上海市市场监管局表示，下一步相关部门将加大网络市场监管力度，重点打击商标侵权、虚假宣传、刷单炒信、价格欺诈等违法行为。",
                "B",
                "电商"
            ],
            [
                "加强网络交易商品（食品）质量抽检，依法查处网络销售不合格商品（食品）行为，严厉打击侵犯公民个人信息等违法犯罪活动。",
                "I",
                "电商"
            ]
        ]
    }
},
}
```
train/val/test.json are data files and each line is a JSON instance. Each JSON instance contains Descriptor and Doc fields, in which Descriptor is event descriptor, and Doc is the annotated document.

### Model Training

#### Step 1: Event-Oriented Opinion Extraction

##### Seq

```python
python eoe_model/seq/main.py \
       --lr 5e-4 \
       --backbone_lr 1e-6 \
       --batch_size 1 \
       --is_eval 0 \
       --bert bert-base-chinese \
       --num_epochs 10 \
       --data_dir data/ECOB-ZH/ \
       --model_dir model_files/chinese_model/ \
       --result_dir result/chinese_result/
```
- ```--bert``` refers to pretrained model path. If you want to train model on English dataset, input 'bert-base-cased'.
- ```--data_dir``` refers to data path. If you want to train model on English dataset, input 'data/ECOB-EN/'.
- ```--model_dir``` refers to the path where the model saved.
- ```--result_dir``` refers to the path where the result saved.

##### PairCls
```python
python eoe_model/paircls/main.py \
       --bert bert-base-chinese \
       --lr 5e-6 \
       --batch_size 24 \
       --retrain 1 \
       --num_epochs 10 \
       --model_folder model_files/chinese_model/ \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/
```

####  Step 2: Opinion Target Extraction
##### MRC
```python
python ote_model/mrc/main.py \
      --model_name_or_path luhua/chinese_pretrain_mrc_roberta_wwm_ext_large \
      --do_train \
      --do_eval \
      --do_lower_case \
      --learning_rate 3e-5 \
      --num_train_epochs 5 \
      --per_gpu_eval_batch_size=4 \
      --per_gpu_train_batch_size=6 \
      --evaluate_during_training \
      --output_dir model_files/chinese_model/mrc/ \
      --data_dir data/ECOB-ZH/ \
      --result_dir result/chinese_result/
```
- ```--model_name_or_path``` refers to pretrained model path. If you want to train model on English dataset, input 'bert-large-uncased-whole-word-masking-finetuned-squad'.

##### SpanR
```python
python ote_model/enum/main.py \
      --lr 1e-5 \
      --batch_size 16 \
      --retrain 1 \
      --bert bert-base-chinese \
      --opinion_level segment \
      --ratio 2 \
      --num_epochs 5 \
      --model_folder model_files/chinese_model/ \
      --data_dir data/ECOB-ZH/ \
      --result_dir result/chinese_result/
```
- ```--ratio``` refers to negative sampling ratio. If you want to train model on English dataset, input '5'.

### Model Evaluation

#### Seperate

##### Seq / PairCls
```python
# Seq
python eval/eval_seq.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model gold

# PairCls
python eval/eval_paircls.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model gold
```
- ```--data_dir``` refers to gold data path. If you want to train model on English dataset, input 'data/ECOB-EN/'.
- ```--result_dir``` refers to the path where the result saved.
- ```--downstream_model``` refers to type of downstream model. Evaluate Seq model on EOT if choose 'gold', evaluate Seq-SpanR/MRC pipeline if 'enum'/'mrc'.

##### MRC / SpanR

```python
# MRC
python ote_model/mrc/main.py \
      --model_name_or_path luhua/chinese_pretrain_mrc_roberta_wwm_ext_large \
      --do_eval \
      --do_lower_case \
      --per_gpu_eval_batch_size=4 \
      --per_gpu_train_batch_size=6 \
      --output_dir model_files/chinese_model/mrc/ \
      --data_dir data/ECOB-ZH/ \
      --result_dir result/chinese_result/ \
      --predict_file test.txt
python eval/eval_aspect.py \
      --language chinese \
      --result_dir result/chinese_result/ \
      --result_file mrc.results \
      --downstream_model mrc

# SpanR
python eval/eval_aspect.py \
      --language chinese \
      --result_dir result/chinese_result/ \
      --result_file enumerate.results \
      --downstream_model enum
```
- ```--language``` refers to the language of data files. If you want to train model on English dataset, input 'english'.
- ```--result_dir``` refers to the directory where the result saved.
- ```--result_dir``` refers to the result file name.
- ```--downstream_model``` refers to type of downstream model. Evaluate Enum/MRC if 'enum'/'mrc'.


#### Pipeline

```python
# PairCls-SpanR/MRC
python eval/eval_paircls.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level segment

# Seq-SpanR/MRC
python eval/eval_seq.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level segment
```
- ```--opinion_level``` refers to segment/sentence-level evaluation metrics. Evaluate by segment/sentence-level if 'segment'/'sent'.

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.