from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import torch
from data_for_pretrain import data_for_pretrain
from tqdm import tqdm
import sentencepiece
from googletrans import Translator

#62100

obj = data_for_pretrain(['ru','zh-cn'])
obj.set_columns()
obj.upload(['source1.tsv','source2.tsv'])
obj.upload(language_name='zh-cn',path=[f'data_for_pretrain_zh-cn{i}.tsv' for i in range(1,4)])
obj.upload(language_name='ru', path=[f'data_for_pretrain_ru{i}.tsv' for i in range(1,4)])

print(obj.data[0].info,obj.data[1].info)

translator = Translator()

for i in tqdm(range(95001,99201)):
    text = obj.raw.values[i,1]
    tranlated = translator.translate(text, dest='ru')
    obj.sentence_uploading('ru',[obj.raw.values[i,0],tranlated.text])

    tranlated = translator.translate(text, dest='zh-cn')
    obj.sentence_uploading('zh-cn', [obj.raw.values[i, 0], tranlated.text])

obj.save(language_name='ru')
obj.save(language_name='zh-cn')

# model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
# tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
# batch = tokenizer(obj.raw.values[0,1].split('.',maxsplit=3), return_tensors="pt",padding=True)
# generated_ids = model.generate(**batch)
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

# url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=%22gene%22AND%22protein%22&retmax' \
#       '=100000&usehistory=y '
# r = requests.get(url)
#
# soup = BeautifulSoup(r.text, features="html.parser")
# arr = soup.find_all('id')
# id_arr = [i.text for i in arr]
#
# print(len(id_arr))
#
# obj = data_for_pretrain(['ru','zh-cn'])
# obj.set_columns()
# for i in tqdm(range(100)):
#     obj.download(ids=id_arr[i*1000:(i+1)*1000])
# print(obj.raw.info)
# obj.save()



