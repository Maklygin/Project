# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from data_for_pretrain import data_for_pretrain
from tqdm import tqdm
import googletrans
from googletrans import Translator



translator = Translator()
obj = data_for_pretrain(['ru','zh-cn'])
obj.set_columns()
obj.upload('source.tsv')


for i in range(10):
    text = obj.raw.values[i,1]
    tranlated = translator.translate(text, dest='ru')
    obj.sentence_uploading('ru',[obj.raw.values[i,0],tranlated.text])
print(obj.data[0].info)

#model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
#tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
#batch = tokenizer([text], return_tensors="pt")
#generated_ids = model.generate(**batch)
#print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

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



