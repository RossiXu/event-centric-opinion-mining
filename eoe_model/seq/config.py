import numpy as np
from tqdm import tqdm
from typing import List
from utils import Instance
import torch
import os
from termcolor import colored
from utils import START, STOP, PAD


class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyper parameters
        """

        # Predefined label string.
        self.PAD = PAD
        self.B = "B"
        self.I = "I"
        self.S = "S"
        self.E = "E"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = "<UNK>"
        self.unk_id = -1

        # Model hyper parameters
        self.embedding_dim = args.embedding_dim
        self.seed = args.seed
        self.hidden_dim = args.hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn

        # Data specification
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.model_dir = args.model_dir
        self.data_dir = args.data_dir
        self.label2idx = {}
        self.idx2labels = []
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num

        # Training hyper parameter
        self.model_folder = args.model_folder
        self.result_dir = args.result_dir
        self.optimizer = args.optimizer.lower()
        self.lr = args.lr
        self.backbone_lr = args.backbone_lr
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)
        self.max_no_incre = args.max_no_incre
        self.bert = args.bert

    def build_word_idx(self, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]) -> None:
        """
        Build the vocab 2 idx for all instances
        :param train_insts:
        :param dev_insts:
        :param test_insts:
        :return:
        """
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = 1
        self.unk_id = 1
        self.idx2word.append(self.UNK)

        self.char2idx[self.PAD] = 0
        self.idx2char.append(self.PAD)
        self.char2idx[self.UNK] = 1
        self.idx2char.append(self.UNK)

        # extract word on train, dev, test
        for inst in train_insts + dev_insts + test_insts:
            for sent in inst.input:
                if sent not in self.word2idx:
                    self.word2idx[sent] = len(self.word2idx)
                    self.idx2word.append(sent)
        # extract char only on train (doesn't matter for dev and test)
        for inst in train_insts:
            for sent in inst.input:
                for c in sent:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)

    def build_label_idx(self, insts: List[Instance]) -> None:
        """
        Build the mapping from label to index and index to labels.
        :param insts: list of instances.
        :return:
        """
        self.label2idx[self.PAD] = len(self.label2idx)
        self.idx2labels.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)

        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        self.label_size = len(self.label2idx)
        print("#labels: {}".format(self.label_size))
        print("label 2idx: {}".format(self.label2idx))

    def use_iobes(self, insts: List[Instance]):
        """
        Use IOBES tagging schema to replace the IOB tagging schema in the instance
        :param insts:
        :return:
        """
        for idx, inst in enumerate(insts):
            output = inst.output
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(self.B):
                        output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        output[pos] = curr_entity.replace(self.I, self.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(self.B):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.I, self.E)
            insts[idx].output = output
        return insts

    def map_insts_ids(self, insts: List[Instance]):
        """
        Create id for word, char and label in each instance.
        """

        for inst in insts:
            sents = inst.input
            inst.word_ids = []
            inst.char_ids = []
            inst.output_ids = [] if inst.output else None
            for sent in sents:
                if sent in self.word2idx:
                    inst.word_ids.append(self.word2idx[sent])
                else:
                    inst.word_ids.append(self.word2idx[self.UNK])
                char_id = []
                for c in sent:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[self.UNK])
                inst.char_ids.append(char_id)
            if inst.output:
                for label in inst.output:
                    inst.output_ids.append(self.label2idx[label])
