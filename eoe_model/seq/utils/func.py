import random

import torch
import numpy as np
from termcolor import colored
from torch import optim, nn

from instance import Instance
from constant import PAD
from typing import List, Tuple


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], 1, vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def batching_list_instances(batch_size, insts: List[Instance], shffule=True):
    """
    List of instances -> List of batches
    """
    if shffule:
        insts.sort(key=lambda x: len(x.input.sents))
    train_num = len(insts)
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(one_batch_insts)
    if shffule:
        random.shuffle(batched_data)
    return batched_data


def simple_batching(config, insts, tokenizer, word_pad_idx=0):

    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        sent_seq_len: Shape: (batch_size), the length of each paragraph in a batch.
        sent_tensor: Shape: (batch_size, max_seq_len, max_token_num)
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_data = insts
    batch_size = len(batch_data)
    word_pad_idx = word_pad_idx
    # doc len
    doc_sents = [inst.input.sents for inst in insts]  # doc_num * doc_len
    events = [inst.event for inst in insts]
    max_sent_len = max([len(doc_sent) for doc_sent in doc_sents])
    sent_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.sents), batch_data)))
    # sent tensor
    doc_sent_ids = []
    max_token_len = 0
    for idx, doc in enumerate(doc_sents):
        sent_ids = [tokenizer.encode_plus(sent, events[idx]).input_ids for sent in doc]  # doc_len * token_num
        max_token_len = max(max_token_len, max([len(sent) for sent in sent_ids]))
        doc_sent_ids.append(sent_ids)
    # padding: batch_size, max_seq_len, max_token_len
    for doc_idx, doc_sent_id in enumerate(doc_sent_ids):
        for sent_idx, sent_id in enumerate(doc_sent_id):
            pad_token_num = - len(sent_id) + max_token_len
            doc_sent_ids[doc_idx][sent_idx].extend([word_pad_idx]*pad_token_num)
        pad_sent_num = max_sent_len - len(doc_sent_id)
        for i in range(pad_sent_num):
            doc_sent_ids[doc_idx].append([word_pad_idx]*max_token_len)
    # label seq tensor
    label_seq_tensor = torch.zeros((batch_size, max_sent_len), dtype=torch.long)
    for idx in range(batch_size):
        if batch_data[idx].output_ids:
            label_seq_tensor[idx, :sent_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)

    # list to tensor
    sent_tensor = torch.LongTensor(doc_sent_ids).to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    sent_seq_len = sent_seq_len.to(config.device)

    return sent_seq_len, sent_tensor, label_seq_tensor


def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def get_optimizer(config, model):
    """
    Method to get optimizer.
    """
    base_params = list(map(id, model.encoder.bert.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": config.lr},
        {"params": model.encoder.bert.parameters(), "lr": config.backbone_lr},
    ]

    if config.optimizer.lower() == "sgd":
        print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params)
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def use_ibo(label):
    if label.startswith('E'):
        label = 'I'
    elif label.startswith('S'):
        label = 'B'
    return label


def write_results(filename: str, insts):
    """
    Save sentence NER result.
    Each line format: sent   gold_label  predicted_label
    """
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        event = inst.event
        title = inst.title
        f.write(event + '\n')
        f.write(title + '\n')
        for i in range(len(inst.input)):
            sents = inst.input.ori_sents
            output = inst.output
            aspect = inst.target
            prediction = inst.prediction
            assert len(output) == len(prediction)
            # sent, pred_label, gold_aspect, gold_label
            f.write("{}\t{}\t{}\t{}\n".format(sents[i], use_ibo(prediction[i]), aspect[i], use_ibo(output[i])))
        f.write("-"*54 + '\n')
    f.close()
