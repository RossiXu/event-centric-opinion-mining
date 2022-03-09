import torch
from tqdm import tqdm
from sentence import Sentence
from instance import Instance
from typing import List
import pickle


class Reader:

    def __init__(self, digit2zero: bool = True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_txt(self, file: str, number: int = 5) -> List[Instance]:
        print("Reading file: " + file)
        insts = []

        # construct instances
        with open(file, 'r', encoding='utf-8') as f:
            sents = []
            ori_sents = []
            labels = []
            targets = []
            sent_idx = 0

            f = f.readlines()
            for line_idx, line in enumerate(tqdm(f)):
                line = line.strip()
                line_split = [ele.strip() for ele in line.strip().split('\t') if len(ele)]
                # skip event and title
                if ('B' not in line and 'I' not in line and 'O' not in line and '-'*54 not in line) \
                        or (len(line_split) != 3 and '-'*54 not in line):
                    if len(line_split) > 1:
                        event = line_split[0]
                    else:
                        title = line
                    continue
                # An instance (there is a passage) end. Store it.
                if '-'*54 in line:
                    if len(sents):
                        inst = Instance(Sentence(sents, ori_sents), event, title, labels, targets)
                        insts.append(inst)

                    sents = []
                    ori_sents = []
                    targets = []
                    labels = []
                    sent_idx = 0
                    if len(insts) == number:
                        break
                    continue
                # Parse a line.
                ls = line_split
                sent, label, target = ls[0], ls[1], ls[2]
                ori_sents.append(sent)
                targets.append(target)

                sent_idx += 1

                sents.append(sent)
                self.vocab.add(sent)

                labels.append(label)
        print("number of documents: {}".format(len(insts)))
        return insts




