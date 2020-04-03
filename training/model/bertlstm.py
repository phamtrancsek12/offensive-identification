"""
This file contains BERT-LSTM model,
which is using BERT model from `transformers` library of huggingface (https://github.com/huggingface/transformers)
and a LSTM layers + FC classifier on top.
As in BERTForSequenceClassification, the loss value is calculated during forward step
and concatenate with the output.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss
from transformers import BertPreTrainedModel
from transformers import BertModel
from config import TRAINING_SCHEME, DEVICE


class BertLSTM(BertPreTrainedModel):
    """BERT-LSTM model"""
    def __init__(self, config):
        super(BertLSTM, self).__init__(config)
        # Get config
        self.num_labels = config.num_labels
        self.batch_size = TRAINING_SCHEME["batch_size"]
        self.bert_hidden_dim = config.hidden_size
        self.lstm_num_layers = TRAINING_SCHEME["lstm_num_layers"]
        self.lstm_hidden_dim = TRAINING_SCHEME["lstm_hidden_dim"]
        self.relu_dim_list = TRAINING_SCHEME["relu_dim_list"]
        self.dropout_prob =  TRAINING_SCHEME["dropout_prob"]
        self.soft_label_ratio = TRAINING_SCHEME["soft_label_ratio"]
        self.max_seq_length = TRAINING_SCHEME["max_seq_length"]

        # BERT
        self.bert = BertModel(config)

        # LSTM
        self.lstm = nn.LSTM(input_size=self.bert_hidden_dim,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_num_layers)

        # FC classifier
        self.hidden2dense = nn.Linear(self.bert_hidden_dim  + self.lstm_hidden_dim, self.relu_dim_list[0])

        modules = []
        for i in range(len(self.relu_dim_list) - 1):
            modules.append(nn.Linear(self.relu_dim_list[i], self.relu_dim_list[i+1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout_prob))
        dense2label = nn.Linear(self.relu_dim_list[-1], self.num_labels)
        modules.append(dense2label)
        self.classifier = nn.Sequential(*modules)

        # ReLU & Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)

        # Init weights
        self.init_weights()
        self.lstm_hidden = self.init_hidden()

    def init_hidden(self):
        """
        Init weights for LSTM layer
        :return: (hidden h, cell c)
        """
        return (Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_dim)).to(DEVICE),
                Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_dim)).to(DEVICE))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, label_ids=None,
                position_ids=None, head_mask=None):
        """
        Forward pass
        :return: output[0] is loss value,
                 output[1] is log_softmax value,
                 the rest are hidden states and attentions
        """
        # Get BERT output
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # Get the value of [CLS]
        x_pool = outputs[1]

        # Get the last hidden-state from BERT output
        # Remove first token [CLS]
        x_sequence = outputs[0]
        x_sequence = x_sequence[:, 1:]
        # Pass the sequence through LSTM layers
        x_sequence = x_sequence.reshape(self.max_seq_length - 1, self.batch_size, -1)
        lstm_out, self.lstm_hidden = self.lstm(x_sequence, self.lstm_hidden)

        # Concat CLS with LSTM output (last hidden values)
        x = torch.cat((x_pool, lstm_out[-1]), dim=1)

        # Pass the output to
        y = self.dropout(x)
        y = self.hidden2dense(x)
        y = self.relu(y)
        y = self.dropout(y)

        logits = self.classifier(y)
        log_softmax = F.log_softmax(logits)

        # Add log_softmax value to outputs
        outputs = (log_softmax,) + outputs[2:]

        # Calculate loss
        if labels is not None:
            if self.num_labels == 1:
                # Loss for regression problem (not use in Offensive task)
                loss_fct = MSELoss()
                loss = loss_fct(log_softmax.view(-1), labels.view(-1))
            else:
                # Loss is the combination of loss on both soft and hard labels
                loss_fct_soft = KLDivLoss()
                loss_fct_hard = CrossEntropyLoss()
                loss = (1 - self.soft_label_ratio) * loss_fct_hard(logits.view(-1, self.num_labels), label_ids.view(-1)) \
                       + self.soft_label_ratio * loss_fct_soft(log_softmax[:, 1].view(-1), labels.view(-1))
        else:
            # For inference phase
            loss = 0

        # Add loss to outputs
        outputs = (loss,) + outputs
        return outputs

