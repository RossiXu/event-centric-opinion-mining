import json

import numpy as np
import pandas as pd
import random
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import jieba


def get_spans(text, language='english'):
    """
    :param text: '高校男生体验女生生理痛'
    :return: '高校', '男生', '体验', '女生', '生理', '痛', '高校男生', '男生体验', '体验女生', '女生生理',
    '生理痛', '高校男生体验', '男生体验女生', '体验女生生理', '女生生理痛', '高校男生体验女生', '男生体验女生生理',
    '体验女生生理痛', '高校男生体验女生生理', '男生体验女生生理痛', '高校男生体验女生生理痛'
    """
    if language == 'chinese':
        words = list(jieba.cut(text))
        spans = []
        for span_len in range(1, len(words) + 1):
            for start_idx in range(0, len(words) - span_len + 1):
                spans.append(''.join(words[start_idx:start_idx + span_len]))
    else:
        words = list(text.split())
        spans = []
        for span_len in range(1, len(words) + 1):
            for start_idx in range(0, len(words) - span_len + 1):
                spans.append(' '.join(words[start_idx:start_idx + span_len]))
    return spans


def data_preprocess(file_name, ratio=3, opinion_level='segment', language='english'):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    insts = []
    spans = {}  # event: spans
    for passage in data:
        event = passage['Descriptor']['text'].strip()
        contents = passage['Doc']['content']
        sents = [content[0].strip() for content in contents]
        labels = [content[1].strip() for content in contents]
        aspects = [content[2].strip() for content in contents]

        if event not in spans.keys():
            spans[event] = get_spans(event, language)

        if opinion_level == 'sent':
            for idx, label in enumerate(labels):
                if label == 'B' or label == 'I':
                    insts.append([sents[idx].strip(), spans[event], event, aspects[idx]])
        elif opinion_level == 'segment':
            opinion = ''
            for idx, label in enumerate(labels):
                if (label == 'B' and not opinion) or label == 'I':
                    opinion += sents[idx]
                elif label == 'B' and opinion:
                    old_aspect = aspects[idx - 1]
                    insts.append([opinion, spans[event], event, old_aspect])
                    opinion = sents[idx]
                elif label == 'O' and opinion:
                    old_aspect = aspects[idx - 1]
                    insts.append([opinion, spans[event], event, old_aspect])
                    opinion = ''
            if opinion:
                insts.append([opinion, spans[event], event, aspects[-1]])

    print(file_name, 'num of opinions:', len(insts), sep='\t')
    input_insts = []
    for inst in insts:
        input_insts.append([inst[0], inst[3], 1, inst[3]])
        choice_spans = random.sample(inst[1], min(ratio, len(inst[1])))
        for choice_span in choice_spans:
            if choice_span != inst[3]:
                input_insts.append([inst[0], choice_span, 0, inst[3]])
    data = pd.DataFrame(input_insts)
    data.columns = ['sent', 'span', 'is_aspect', 'gold_aspect']
    return data


def lr_decay(learning_rate, lr_decay_metric, optimizer, epoch):
    lr = learning_rate / (1 + lr_decay_metric * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def use_ibo(label):
    if label.startswith('E-'):
        label = 'I-' + label[2:]
    elif label.startswith('S-'):
        label = 'B-' + label[2:]
    return label


def get_important_sent(text, tf_idf, max_length=430):
    curr_length = 0
    for line in text:
        curr_length += len(line.strip().split())
    if curr_length <= max_length:
        return text

    important_text = []
    curr_length = 0

    vocab = {value: key for key, value in tf_idf.vocabulary_.items()}

    doc_tf_idf = tf_idf.transform([' '.join(text)]).toarray()
    keywords_idx = np.argsort(-doc_tf_idf).tolist()[0]
    for keyword_idx in keywords_idx:
        keyword = vocab[keyword_idx]
        for line in text:
            if keyword in line:
                important_text.append(line)
                curr_length += len(line.strip().split())
                if curr_length > max_length:
                    return important_text
    return important_text


def data_batch(data_df, batch_size, shuffle=True):
    # Tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # Get data.
    text_a = list(data_df.loc[:, 'sent'])
    text_b = list(data_df.loc[:, 'span'])
    labels = list(data_df.loc[:, 'is_aspect'])
    gold_aspect = list(data_df.loc[:, 'gold_aspect'])

    # Shuffle.
    inputs = list(zip(text_a, text_b, labels, gold_aspect))
    if shuffle:
        inputs = sorted(inputs, key=lambda x: len(x[0] + x[1]))

    batches = []
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:min(i + batch_size, len(inputs) + 1)]
        batch_text_a, batch_text_b, batch_labels, batch_gold_aspect = zip(*batch)
        tokenized_input = tokenizer(batch_text_a, batch_text_b, max_length=512, padding='max_length', truncation='longest_first')
        input_ids, token_type_ids, attention_mask = tokenized_input.input_ids, \
                                                    tokenized_input.token_type_ids, \
                                                    tokenized_input.attention_mask
        batch_input = (torch.LongTensor(input_ids),
                       torch.LongTensor(token_type_ids),
                       torch.LongTensor(attention_mask),
                       torch.LongTensor(batch_labels),
                       batch_text_a,
                       batch_text_b,
                       batch_gold_aspect)
        batches.append(batch_input)
    if shuffle:
        random.shuffle(batches)
    return batches