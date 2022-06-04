import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer,AdamW
from model_script import BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.nn import CrossEntropyLoss

from model_script import convert_tsv_to_model_input, convert_list_to_torch

if __name__ == '__main__':
    num_labels = 6
    max_seq_length = 256
    batch_size = 16
    num_epoch = 1
    lr = 2.5e-5
    label_map = {'false': 0, 'CPR:3': 1, 'CPR:4': 2, 'CPR:5': 3, 'CPR:6': 4, 'CPR:9': 5}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=num_labels)

    df = pd.read_csv('train_en.tsv', index_col=False)

    model_input_list = convert_tsv_to_model_input('train_en.tsv', tokenizer=tokenizer, max_seq_length=256)

    train_data = TensorDataset(convert_list_to_torch(model_input_list))
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    model.train()
    for i in tqdm(range(num_epoch)):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)

            title_ids, title_mask, title_segment, input_ids, input_mask, \
            segment_ids, P_gauss1_list, P_gauss2_list, label_ids = batch

            logits = model(title_ids, title_segment, title_mask, input_ids, segment_ids,
                           input_mask, P_gauss1_list, P_gauss2_list, labels=None)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))