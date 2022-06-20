import json

import numpy as np
import pandas as pd
import random
import torch
from termcolor import colored
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
    print("Reading " + file_name + " file.")

    # Read document.
    with open(file_name + '.doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # Read annotation.
    try:
        with open(file_name + '.ann.json', 'r', encoding='utf-8') as f:
            opinions = json.load(f)
    except FileNotFoundError:
        opinions = []
        print(colored('[There is no ' + file_name + '.ann.json.]', 'red'))

    spans = {}  # event: spans
    insts = []
    for doc in docs:
        event = doc['Descriptor']['text']
        event_id = doc['Descriptor']['event_id']
        doc_id = doc['Doc']['doc_id']
        contents = doc['Doc']['content']
        sents = [content['sent_text'] for content in contents]

        if event not in spans.keys():
            spans[event] = get_spans(event, language)

        doc_opinions = [opinion for opinion in opinions if int(opinion['doc_id']) == int(doc_id)]
        for doc_opinion in doc_opinions:
            opinion_sents = sents[doc_opinion['start_sent_idx']:doc_opinion['end_sent_idx']+1]
            opinion_text = ' '.join(opinion_sents)
            try:
                argument = doc_opinion['argument']
            except KeyError:
                print(colored('[There is no gold argument!', 'red'))
                argument = ''

            inst = [opinion_text, spans[event], event, argument, event_id, doc_id, doc_opinion['start_sent_idx'], doc_opinion['end_sent_idx']]
            insts.append(inst)

    print(file_name, 'Number of opinions:', len(insts), sep='\t')
    input_insts = []
    for inst in insts:
        input_insts.append([inst[0], inst[3], 1, inst[3], inst[4], inst[5], inst[6], inst[7]])
        choice_spans = random.sample(inst[1], min(ratio, len(inst[1])))
        for choice_span in choice_spans:
            if choice_span != inst[3]:
                input_insts.append([inst[0], choice_span, 0, inst[3], inst[4], inst[5], inst[6], inst[7]])
    data = pd.DataFrame(input_insts)
    data.columns = ['sent', 'span', 'is_aspect', 'gold_aspect', 'event_id', 'doc_id', 'start_sent_idx', 'end_sent_idx']
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
    event_id = list(data_df.loc[:, 'event_id'])
    doc_id = list(data_df.loc[:, 'doc_id'])
    start_sent_idx = list(data_df.loc[:, 'start_sent_idx'])
    end_sent_idx = list(data_df.loc[:, 'end_sent_idx'])

    # Shuffle.
    inputs = list(zip(text_a, text_b, labels, gold_aspect, event_id, doc_id, start_sent_idx, end_sent_idx))
    if shuffle:
        inputs = sorted(inputs, key=lambda x: len(x[0] + x[1]))

    batches = []
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:min(i + batch_size, len(inputs) + 1)]
        batch_text_a, batch_text_b, batch_labels, batch_gold_aspect, batch_event_id, batch_doc_id, batch_start_sent_idx, batch_end_sent_idx = zip(*batch)
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
                       batch_gold_aspect,
                       batch_event_id,
                       batch_doc_id,
                       batch_start_sent_idx,
                       batch_end_sent_idx)
        batches.append(batch_input)
    if shuffle:
        random.shuffle(batches)
    return batches