# 代码结构

```
event-centric-opinion-mining
├── data
├── ── ECOB-ZH  # 中文数据集
├── ── ECOB-EN  # 英文数据集
├── data_preprocess
├── ── split_data.py
├── ── stastic.py
├── eval
├── ── eval_aspect.py
├── ── eval_paircls.py
├── ── eval_seq.py
├── eoe_model  # 提取与事件相关的观点片段
├── ── paircls  # 输入(event, sentence)句对到BERT，提取与事件相关的观点句
├── ── seq  # 输入(event, passage)到BERT+BiLSTM+CRF，提取与事件相关的观点片段
├── ote_model  # 提取观点片段对应的观点对象
├── ── enum  # 枚举
├── ── mrc  # 机器阅读理解
└── requirment.txt
```
# 环境配置

```python
conda create -n ecom python=3.8
conda activate ecom

pip install -r requirements.txt
```

# 运行流程

## 数据及统计

将数据解压缩放至data/。

运行以下代码分别获得中英文数据集的统计数据，对应Table1.

```python
# 中文数据集统计数据
python data_preprocess/stastic.py --data_dir data/ECOB-ZH/ --language chinese

# 英文数据集统计数据
python data_preprocess/stastic.py --data_dir data/ECOB-EN/ --language english
```

## 提取与事件相关的观点片段
运行以下代码提取与事件相关的观点片段，对应Table 3。

1. 使用序列标注方法（Seq）：

```python
# 中文数据集
# 训练模型
python eoe_model/seq/main.py \
       --lr 5e-4 \
       --backbone_lr 1e-6 \
       --batch_size 1 \
       --is_eval 0 \
       --bert bert-base-chinese \
       --num_epochs 10 \
       --model_dir model_files/chinese_model/ \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/
# 结果评价
python eval/eval_seq.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model gold

# 英文数据集
# 训练模型
python eoe_model/seq/main.py \
       --lr 5e-4 \
       --backbone_lr 1e-6 \
       --batch_size 1 \
       --is_eval 0 \
       --bert bert-base-cased \
       --num_epochs 10 \
       --model_dir model_files/english_model/ \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/
# 结果评价
python eval/eval_seq.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model gold
```

2. 使用句对关系分类方法（PairCls）：

```python
# 中文数据集
# 训练模型
python eoe_model/paircls/main.py \
       --bert bert-base-chinese \
       --lr 5e-6 \
       --batch_size 24 \
       --retrain 1 \
       --num_epochs 10 \
       --model_folder model_files/chinese_model/ \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/
# 结果评价
python eval/eval_paircls.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model gold

# 英文数据集
# 训练模型
python eoe_model/paircls/main.py \
       --bert bert-base-cased \
       --lr 5e-6 \
       --batch_size 24 \
       --retrain 1 \
       --num_epochs 10 \
       --model_folder model_files/english_model/ \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/
# 结果评价
python eval/eval_paircls.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model gold
```

## 观点片段对应的观点对象提取

运行以下代码提取观点片段对应的观点对象，对应Table 4。

1. 使用机器阅读理解方法（MRC）：

```python
# 中文数据集
# 模型训练
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
# 模型测试
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
# 模型评估
python eval/eval_aspect.py \
      --language chinese \
      --result_dir result/chinese_result/ \
      --result_file mrc.results \
      --downstream_model mrc

# 英文数据集
# 模型训练
python ote_model/mrc/main.py \
      --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
      --do_train \
      --do_eval \
      --do_lower_case \
      --learning_rate 5e-6 \
      --num_train_epochs 5 \
      --per_gpu_eval_batch_size=4 \
      --per_gpu_train_batch_size=6 \
      --evaluate_during_training \
      --output_dir model_files/english_model/mrc/ \
      --data_dir data/ECOB-EN/ \
      --result_dir result/english_result/
# 模型测试
python ote_model/mrc/main.py \
      --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
      --do_eval \
      --do_lower_case \
      --per_gpu_eval_batch_size=4 \
      --per_gpu_train_batch_size=6 \
      --output_dir model_files/english_model/mrc/ \
      --data_dir data/ECOB-EN/ \
      --result_dir result/english_result/ \
      --predict_file test.txt
# 模型评估
python eval/eval_aspect.py \
      --language english \
      --result_dir result/english_result/ \
      --result_file mrc.results \
      --downstream_model mrc \
      --data_dir data/ECOB-EN/
```

2. 使用枚举方法（Enum）：

```python
# 中文数据集
# 模型训练
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
# 模型评估
python eval/eval_aspect.py \
      --language chinese \
      --result_dir result/chinese_result/ \
      --result_file enumerate.results \
      --downstream_model enum

# 英文数据集
# 模型训练
python ote_model/enum/main.py \
      --lr 1e-5 \
      --batch_size 16 \
      --retrain 1 \
      --bert bert-base-cased \
      --opinion_level segment \
      --ratio 5 \
      --num_epochs 5 \
      --model_folder model_files/english_model/ \
      --data_dir data/ECOB-EN/ \
      --result_dir result/english_result/
# 模型评估
python eval/eval_aspect.py \
      --data_dir data/ECOB-EN/ \
      --language english \
      --result_dir result/english_result/ \
      --result_file enumerate.results \
      --downstream_model enum
```
## Pipeline
运行以下代码得到pipeline运行结果，对应Table 2。
```python
# 中文数据集

# PairCls-SpanR
# Segment level
python eval/eval_paircls.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level segment
# Sentence level
python eval/eval_paircls.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level sent

# PairCls-MRC
# Segment level
python eval/eval_paircls.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level segment
# Sentence level
python eval/eval_paircls.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level sent

# Seq-SpanR
# Segment level
python eval/eval_seq.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level segment
# Sentence level
python eval/eval_seq.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level sent

# Seq-MRC
# Segment level
python eval/eval_seq.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level segment
# Sentence level
python eval/eval_seq.py \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level sent

# 英文数据集
# 同上
# 将data_dir替换成为 data/ECOB-EN/
# 将result_dir 替换成为 result/english_results/
# 将model_folder 替换成为 model_files/english_model/
# 将涉及的预训练模型替换成为英文版本

# PairCls-SpanR
# Segment level
python eval/eval_paircls.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level segment
# Sentence level
python eval/eval_paircls.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level sent

# PairCls-MRC
# Segment level
python eval/eval_paircls.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level segment
# Sentence level
python eval/eval_paircls.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level sent

# Seq-SpanR
# Segment level
python eval/eval_seq.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level segment
# Sentence level
python eval/eval_seq.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model enum \
       --downstream_file enumerate.results \
       --opinion_level sent

# Seq-MRC
# Segment level
python eval/eval_seq.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level segment
# Sentence level
python eval/eval_seq.py \
       --data_dir data/ECOB-EN/ \
       --result_dir result/english_result/ \
       --downstream_model mrc \
       --downstream_file mrc.results \
       --opinion_level sent
```

## 分析
### 探究不同观点长度对提取结果的影响
运行以下代码得到不同观点长度对提取结果的影响，对应Figure 3。

```python
# 中文数据集
python eval/analysis_opinion_len.py \
        --result_dir result/chinese_result/ \
        --result_file seq.results \
        --data_dir data/ECOB-ZH/

# 英文数据集
python eval/analysis_opinion_len.py \
        --result_dir result/english_result/ \
        --result_file seq.results \
        --data_dir data/ECOB-EN/ \
        --language english
```
### 探究不同观点对象类型对提取结果的影响
```python
# 中文数据集
# 更换aspect_level，分别获得subevent/event/entity类型的观点句的提取结果，这里用Accuracy代表
# subevent
python eval/eval_aspect.py \
        --data_dir data/ECOB-ZH/ \
        --result_dir result/chinese_result/ \
        --downstream_model mrc \
        --result_file mrc.results \
        --aspect_level subevent
# entity
python eval/eval_aspect.py \
        --data_dir data/ECOB-ZH/ \
        --result_dir result/chinese_result/ \
        --downstream_model mrc \
        --result_file mrc.results \
        --aspect_level entity
# event
python eval/eval_aspect.py \
        --data_dir data/ECOB-ZH/ \
        --result_dir result/chinese_result/ \
        --downstream_model mrc \
        --result_file mrc.results \
        --aspect_level event

# 英文数据集
# 更换aspect_level，分别获得subevent/event/entity类型的观点句的提取结果，这里用Accuracy代表
# subevent
python eval/eval_aspect.py \
        --data_dir data/ECOB-EN/ \
        --result_dir result/english_result/ \
        --downstream_model mrc \
        --result_file mrc.results \
        --aspect_level subevent
# entity
python eval/eval_aspect.py \
        --data_dir data/ECOB-EN/ \
        --result_dir result/english_result/ \
        --downstream_model mrc \
        --result_file mrc.results \
        --aspect_level event
# event
python eval/eval_aspect.py \
        --data_dir data/ECOB-EN/ \
        --result_dir result/english_result/ \
        --downstream_model mrc \
        --result_file mrc.results \
        --aspect_level entity
```