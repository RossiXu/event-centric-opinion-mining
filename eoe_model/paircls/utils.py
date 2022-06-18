import json

import numpy as np
import pandas as pd
import random
import torch
from termcolor import colored
from transformers import BertTokenizer
from tqdm import tqdm


def read_data(file_name):
    print("Reading " + file_name + " file.")
    insts = []

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

    pairs = []
    for doc in docs:
        event = doc['Descriptor']['text']
        event_id = doc['Descriptor']['event_id']
        doc_id = doc['Doc']['doc_id']
        contents = doc['Doc']['content']
        sents = [content['sent_text'] for content in contents]
        labels = ['O'] * len(sents)
        doc_opinions = [opinion for opinion in opinions if int(opinion['doc_id']) == int(doc_id)]
        for opinion in doc_opinions:
            for sent_idx in range(opinion['start_sent_idx'], opinion['end_sent_idx'] + 1):
                labels[sent_idx] = 'I'
            labels[opinion['start_sent_idx']] = 'B'
        for sent_idx, sent in enumerate(sents):
            label = 0 if labels[sent_idx] == 'O' else 1
            pairs.append([sent, event, label, event_id, doc_id, sent_idx])

    data = pd.DataFrame(pairs)
    data.columns = ['sent', 'event', 'is_opinion', 'event_id', 'doc_id', 'sent_id']

    print("Number of sentences: {}".format(len(data)))
    print("Number of opinion sentences: {}".format(sum([pair[2] for pair in pairs])))
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


def data_batch(data_df, batch_size, raw_model, shffule=True):
    # Tokenizer.
    tokenizer = BertTokenizer.from_pretrained(raw_model)

    # Get data.
    text_a = list(data_df.loc[:, 'sent'])
    text_b = list(data_df.loc[:, 'event'])
    labels = list(data_df.loc[:, 'is_opinion'])
    event_ids = list(data_df.loc[:, 'event_id'])
    doc_ids = list(data_df.loc[:, 'doc_id'])
    sent_ids = list(data_df.loc[:, 'sent_id'])

    # Shuffle.
    inputs = list(zip(text_a, text_b, labels, event_ids, doc_ids, sent_ids))
    if shffule:
        inputs = sorted(inputs, key=lambda x: len(x[0] + x[1]))

    batches = []
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:min(i + batch_size, len(inputs) + 1)]
        batch_text_a, batch_text_b, batch_labels, batch_event_ids, batch_doc_ids, batch_sent_ids = zip(*batch)
        tokenized_input = tokenizer(batch_text_a, batch_text_b, max_length=512, padding=True, truncation=True)
        input_ids, token_type_ids, attention_mask = tokenized_input.input_ids, \
                                                    tokenized_input.token_type_ids, \
                                                    tokenized_input.attention_mask
        batch_input = (torch.LongTensor(input_ids),
                       torch.LongTensor(token_type_ids),
                       torch.LongTensor(attention_mask),
                       torch.LongTensor(batch_labels),
                       batch_text_a,
                       batch_text_b,
                       batch_event_ids,
                       batch_doc_ids,
                       batch_sent_ids)
        batches.append(batch_input)
    if shffule:
        random.shuffle(batches)
    return batches


def get_opinions(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.read()
    psgs = data.split('-' * 54)
    insts = []
    for psg in psgs:
        psg = psg.strip().split('\n')
        event = psg[0].split('\t')[0]
        lines = psg[2:]
        opinion = []
        for line_idx, line in enumerate(lines):
            if len(line.strip().split('\t')) == 3:
                sent, label, aspect = [ele for ele in line.strip().split('\t') if len(ele)]
            else:
                continue
            if (label == 'B' and not opinion) or label == 'I':
                opinion.append(sent)
            elif label == 'B' and opinion:
                old_aspect = [ele for ele in lines[line_idx - 1].strip().split('\t') if len(ele)][2]
                insts.append([opinion, event, old_aspect])
                opinion = [sent]
            elif label == 'O' and opinion:
                old_aspect = [ele for ele in lines[line_idx-1].strip().split('\t') if len(ele)][2]
                insts.append([opinion, event, old_aspect])
                opinion = []
        if len(opinion):
            insts.append([opinion, event, aspect])
    insts = [inst for inst in insts if '其他' != inst[2]]
    print(file_name, 'num of opinions:', len(insts), sep='\t')
    return insts


def evaluate_f():
    pred_opinions = []
    with open('../../data/pair_classification.results', 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        contents = []
        for line in lines:
            if int(line[3]) == 1:
                contents.append(line[1].strip())
                continue
            elif int(line[3]) == 0 and len(contents):
                pred_opinions.append(contents)
                contents = []
                continue
    gold_opinions = get_opinions('../../data/ECOB-ZH/test.txt')
    gold_opinions = [opinion[0] for opinion in gold_opinions]
    correct_opinions = []
    for pred_opinion in pred_opinions:
        for gold_opinion in gold_opinions:
            if set(pred_opinion) == set(gold_opinion):
                correct_opinions.append(pred_opinion)
                continue

    print('len of pred: ', len(pred_opinions))
    print('len of gold: ', len(gold_opinions))
    print('len of corr: ', len(correct_opinions))
    p = len(correct_opinions) / len(pred_opinions)
    r = len(correct_opinions) / len(gold_opinions)
    f = (2*p*r) / (p+r)
    print(p, r, f)


if __name__ == '__main__':
    evaluate_f()