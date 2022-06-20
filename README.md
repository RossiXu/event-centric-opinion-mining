# Event-Centric Opinion Mining
- An implementation for ECO v1: Towards Event-Centric Opinion Mining.
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
**Additional Statement：** We organize [an evaluation](http://e-com.ac.cn/ccl2022.html/) in CCL2022. So the annoation in test data is not available currently. After the evaluation, we'll open full dataset.

Data folder contains two folders: ECOB-EN and ECOB-ZH.

Before training models, you should first download [data](http://123.57.148.143:9876/down/mRVUaxtM7oUz) and unzip them as follows. 
```
data
├── ECOB-ZH  # Chinese dataset.
├── ── train.doc.json
├── ── train.ann.json
├── ── dev.doc.json
├── ── dev.ann.json
├── ── test.doc.json
├──   ECOB-EN  # English dataset.
├── ── train.doc.json
├── ── train.ann.json
├── ── dev.doc.json
├── ── dev.ann.json
└── ── test.doc.json
```

The data format is as follows:

In train/dev/test.doc.json, each JSON instance represents a document.
```
{
    "Descriptor": {
        "event_id": (int) event_id,
        "text": "Event descriptor."
    },
    "Doc": {
        "doc_id": (int) doc_id,
        "title": "Title of document.",
        "content": [
            {
                "sent_idx": 0,
                "sent_text": "Raw text of the first sentence."
            },
            {
                "sent_idx": 1,
                "sent_text": "Raw text of the second sentence."
            },
            ...
            {
                "sent_idx": n-1,
                "sent_text": "Raw text of the (n-1)th sentence."
            }
        ]
    }
}
```

In train/dev/test.ann.json, each JSON instance represents an opinion extracted from documents.
```
[
	{
            "event_id": (int) event_id,
            "doc_id": (int) doc_id,
            "start_sent_idx": (int) "Sent idx of first sentence of the opinion.",
            "end_sent_idx": (int) "Sent idx of last sentence of the opinion.",
            "argument": (str) "Event argument (opinion target) of the opinion."
  	}
]
```

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

```python
python eval/eval.py \
      --gold_file data/ECOB-ZH/test.ann.json \
      --pred_file result/chinese_result/pred.ann.json
```
## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.
