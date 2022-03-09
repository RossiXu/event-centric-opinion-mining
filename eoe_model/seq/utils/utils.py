import random

import torch
import numpy as np
from termcolor import colored
from torch import optim, nn

from utils import Instance, PAD
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


def batching_list_instances(batch_size, insts: List[Instance]):
    """
    List of instances -> List of batches
    """
    train_num = len(insts)
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(one_batch_insts)

    return batched_data


def simple_batching(config, insts: List[Instance]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        sent_seq_len: Shape: (batch_size), the length of each paragraph in a batch.
        init_sent_emb_tensor: Shape: (batch_size, max_seq_len, emb_size)
        char_seq_tensor: Shape: (batch_size, max_seq_len, max_char_seq_len)
        char_seq_len: Shape: (batch_size, max_seq_len)
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_size = len(insts)
    batch_data = insts
    sent_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.sents), batch_data)))
    max_seq_len = sent_seq_len.max()

    num_tokens = list(map(lambda inst: inst.num_tokens, batch_data))  # 2-dimension
    max_tokens = max([max(num_token) for num_token in num_tokens])

    # NOTE: Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.sents)) + [1] * (int(max_seq_len) - len(inst.input.sents)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    emb_size = 768

    label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)
    initial_sent_emb_tensor = torch.zeros((batch_size, max_seq_len, max_tokens, emb_size), dtype=torch.float32)

    num_tokens_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    for idx in range(batch_size):

        if batch_data[idx].output_ids:
            label_seq_tensor[idx, :sent_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
            num_tokens_tensor[idx, :sent_seq_len[idx]] = torch.LongTensor(batch_data[idx].num_tokens)

        for sent_idx in range(sent_seq_len[idx]):
            for token_idx in range(num_tokens[idx][sent_idx]):
                initial_sent_emb_tensor[idx, sent_idx, token_idx, :emb_size] = batch_data[idx].vec[sent_idx][token_idx]
            char_seq_tensor[idx, sent_idx, :char_seq_len[idx, sent_idx]] = torch.LongTensor(batch_data[idx].char_ids[sent_idx])

        for sentIdx in range(sent_seq_len[idx], max_seq_len):
            # because line 119 makes it 1, every single character should have a id. but actually 0 is enough
            char_seq_tensor[idx, sentIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])

    label_seq_tensor = label_seq_tensor.to(config.device)
    char_seq_tensor = char_seq_tensor.to(config.device)
    sent_seq_len = sent_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)

    return sent_seq_len, num_tokens_tensor, initial_sent_emb_tensor, char_seq_tensor, char_seq_len,  label_seq_tensor


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


def get_optimizer(config, model: nn.Module):
    """
    Method to get optimizer.
    """
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(
            colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
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


def write_results(filename: str, insts):
    """
    Save sentence NER result.
    Each line format: sent   gold_label  predicted_label
    """
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.input)):
            sents = inst.input.ori_sents
            output = inst.output
            prediction = inst.prediction
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\n".format(sents[i], output[i], prediction[i]))
        f.write("\n")
    f.close()
