import torch.nn as nn
import torch
from transformers import BertModel

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BertLSTMEncoder(nn.Module):

    def __init__(self, config):
        super(BertLSTMEncoder, self).__init__()
        # parameters
        self.num_layers = 1
        self.label_size = config.label_size
        self.input_size = config.embedding_dim

        self.device = config.device

        self.label2idx = config.label2idx
        self.labels = config.idx2labels

        final_hidden_dim = config.hidden_dim

        # model
        self.bert = BertModel.from_pretrained(config.bert).to(self.device)

        self.lstm = nn.LSTM(self.input_size,
                            config.hidden_dim // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True).to(self.device)

        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)

        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)

    def forward(self, sent_seq_lens, sent_tensor):
        batch_sent = sent_tensor
        # sentence embedding
        batch_sent = batch_sent[:, :, :420]  # (batch_size, max_sent_len, max_num_len)
        batch_sent_flatten = batch_sent.view(-1, batch_sent.shape[2])  # (batch_size*max_sent_len, max_num_len)
        try:
            batch_sent_output = self.bert(batch_sent_flatten, attention_mask=batch_sent_flatten.gt(0))[0]  # (batch_size*max_seq_len, max_num_len, hidden_size)
        except RuntimeError:
            print(batch_sent.shape)
        batch_sent_output = batch_sent_output[:, 0, :].view(batch_sent.shape[0], batch_sent.shape[1], -1)  # (batch_size, max_seq_len, hidden_size)

        # Sentence LSTM
        sent_rep = self.word_drop(batch_sent_output)
        sorted_seq_len, permIdx = sent_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = sent_rep[permIdx]
        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)

        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        feature_out = self.drop_lstm(lstm_out)

        outputs = self.hidden2tag(feature_out)

        return feature_out[recover_idx], outputs[recover_idx]

