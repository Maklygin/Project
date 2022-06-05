import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss

import pandas as pd
import scipy.stats as st
import numpy as np


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification,self).__init__(config)
        self.num_labels = 6
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.activation = nn.ReLU
        self.activation_final = nn.Tanh()
        self.classifier = nn.Linear(4 * config.hidden_size, self.num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, title_ids, title_type_ids=None, title_mask=None, input_ids=None, token_type_ids=None,
                attention_mask=None, P_gauss1_bs=None, P_gauss2_bs=None, labels=None):

        input_token_output_, pooled_output_ = self.bert(input_ids, token_type_ids, attention_mask)
        input_token_output_ = self.dropout(input_token_output_)

        # input_token_output_ = self.activation(input_token_output_)

        _, title_output = self.bert(title_ids, title_type_ids, title_mask)
        title_query = title_output.unsqueeze(dim=1)
        t_u = torch.matmul(title_query, input_token_output_.transpose(-1, -2))
        t_alpha = nn.Softmax(dim=-1)(t_u)
        t_v = torch.matmul(t_alpha, input_token_output_)
        output_t = t_v.squeeze(dim=1)
        output_t = self.activation_final(output_t)
        output_t = self.dropout(output_t)

        P_gauss1_bs = P_gauss1_bs.unsqueeze(dim=1)
        P_gauss2_bs = P_gauss2_bs.unsqueeze(dim=1)
        gauss_entity1 = torch.matmul(P_gauss1_bs, input_token_output_)
        gauss_entity2 = torch.matmul(P_gauss2_bs, input_token_output_)
        gauss_entity1 = gauss_entity1.squeeze(dim=1)
        gauss_entity2 = gauss_entity2.squeeze(dim=1)
        gauss_entity1 = self.activation_final(gauss_entity1)
        gauss_entity2 = self.activation_final(gauss_entity2)
        gauss_entity1 = self.dropout(gauss_entity1)
        gauss_entity2 = self.dropout(gauss_entity2)

        pooled_output_ = self.dropout(pooled_output_)
        output = torch.cat((output_t, pooled_output_, gauss_entity1, gauss_entity2), -1)
        output = self.dropout(output)
        logits = self.classifier(output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def convert_tsv_to_model_input(path, tokenizer, max_seq_length):
    mu = 0.
    sigma = 2.5
    norm_ = st.norm(mu, sigma).cdf([0, 1, 2, 3, 4, 5, 6, 7])

    model_input = []

    df = pd.read_csv(path, index_col=False)
    title_list, article_list, label_list = [df.values[:, i] for i in [1, 2, 3]]

    label_map = {'false': 0, 'CPR:3': 1, 'CPR:4': 2, 'CPR:5': 3, 'CPR:6': 4, 'CPR:9': 5}

    for i in range(len(title_list)):

        title = tokenizer.tokenize(title_list[i]) \
            if len(tokenizer.tokenize(title_list[i])) <= max_seq_length - 2 \
            else tokenizer.tokenize(title_list[i])[:max_seq_length - 2]

        article = tokenizer.tokenize(article_list[i]) \
            if len(tokenizer.tokenize(article_list[i])) <= max_seq_length - 2 \
            else tokenizer.tokenize(article_list[i])[:max_seq_length - 2]

        title = ["[CLS]"] + title + ["[SEP]"]
        article = ["[CLS]"] + article + ["[SEP]"]

        first_at = article.index('@')
        if article[first_at + 1] == 'chemical':
            chem_index = first_at
            gene_index = article.index('@', first_at + 1)
        else:
            gene_index = first_at
            try:
                chem_index = article.index('@', first_at + 1)
            except:
                print(article)          # Если отсутствует @CHEMICAL или @GENE
                print(df.values[:,0][i])

        # Первое гауссово распредление относится к chem, а второе к gene
        P_gauss1_array = np.zeros(max_seq_length)
        P_gauss2_array = np.zeros(max_seq_length)

        for index, arr in zip((chem_index, gene_index), (P_gauss1_array, P_gauss2_array)):
            j = 0
            while index + j < max_seq_length and j < 7:
                arr[index + j] = norm_[j + 1] - norm_[j]
                j += 1
            j = 1
            while index - j > 0 and j < 7:
                arr[index - j] = norm_[j + 1] - norm_[j]
                j += 1

        P_gauss1_list = list(P_gauss1_array)
        P_gauss2_list = list(P_gauss2_array)

        title_segment = [0] * len(title)
        segment_ids = [0] * len(article)

        title_ids = tokenizer.convert_tokens_to_ids(title)
        input_ids = tokenizer.convert_tokens_to_ids(article)

        title_mask = [1] * len(title_ids)
        input_mask = [1] * len(input_ids)

        padding_title = [0] * (max_seq_length - len(title_ids))
        title_ids += padding_title

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding

        title_mask += padding_title
        input_mask += padding

        title_segment += padding_title
        segment_ids += padding

        assert len(title_ids) == max_seq_length
        assert len(title_mask) == max_seq_length
        assert len(title_segment) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(P_gauss1_list) == max_seq_length
        assert len(P_gauss2_list) == max_seq_length

        label = label_map[label_list[i]]

        model_input.append([title_ids, title_mask, title_segment,
                            input_ids, input_mask, segment_ids,
                            P_gauss1_list, P_gauss2_list, label])

    return model_input


def convert_list_to_torch(model_input):
    all_title_ids = torch.tensor([i[0] for i in model_input], dtype=torch.long)
    all_title_mask = torch.tensor([i[1] for i in model_input], dtype=torch.long)
    all_title_segment = torch.tensor([i[2] for i in model_input], dtype=torch.long)

    all_input_ids = torch.tensor([i[3] for i in model_input], dtype=torch.long)
    all_input_mask = torch.tensor([i[4] for i in model_input], dtype=torch.long)
    all_segment_ids = torch.tensor([i[5] for i in model_input], dtype=torch.long)

    all_P_gauss1_list = torch.tensor([i[6] for i in model_input], dtype=torch.float)
    all_P_gauss2_list = torch.tensor([i[7] for i in model_input], dtype=torch.float)

    all_label_ids = torch.tensor([i[8] for i in model_input], dtype=torch.long)

    return all_title_ids, all_title_mask, all_title_segment, all_input_ids, all_input_mask, \
           all_segment_ids, all_P_gauss1_list, all_P_gauss2_list, all_label_ids
