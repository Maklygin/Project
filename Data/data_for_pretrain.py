import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np


class data_for_pretrain:
    def __init__(self, languages: list):
        self.columns = None
        arr = [(languages[i], i) for i in range(len(languages))]
        self.languages = dict(arr)

        self.data = None
        self.raw = None

    def set_columns(self, columns=['id', 'abstract']):
        # Закрепляем названия колонок
        self.columns = dict.fromkeys(columns)

        frame = pd.DataFrame(columns=self.columns)
        self.data = [frame for i in self.languages]
        self.raw = frame

    def download(self, ids: list):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='
        _url = '&api_key=6fa962de9df48c1cf39c2196ed5bff499e09&rettype=txt'

        ids = np.array(ids)
        ids = ids.reshape((int(len(ids) / 10),10))
        # batch size = 10

        for batch in ids:
            url_id = ''
            for j in batch:
                url_id += j + ','

            r = requests.get(url + url_id + _url)
            soup = BeautifulSoup(r.text, features="html.parser")
            abstracts = soup.find_all('abstract')
            j = 0
            if len(abstracts) == 10:
                for parts in abstracts:
                    text = ''
                    for i in parts.find_all('abstracttext'):
                        text += i.text + ' '

                    # (id=batch[j],abstract=text)
                    df1 = pd.DataFrame(columns=self.raw.columns, data=self.dict_fill([batch[j], text]),index=[0])
                    self.raw = pd.concat([self.raw, df1],ignore_index=True)
                    j += 1
            else:
                for element in batch:
                    r = requests.get(url+str(element)+_url)
                    soup = BeautifulSoup(r.text, features="html.parser")
                    abstract = soup.find('abstract')
                    if abstract is not None:
                        text = ''
                        for i in abstract.find('abstracttext'):
                            text += i.text + ' '
                        df1 = pd.DataFrame(columns=self.raw.columns, data=self.dict_fill([batch[j], text]), index=[0])
                        self.raw = pd.concat([self.raw, df1], ignore_index=True)
                    j += 1

    def dict_fill(self, x: list):
        j = 0
        dict_ = self.columns.copy()
        for i in self.columns:
            dict_[i] = x[j]
            j += 1
        return dict_

    def show(self, language_name=None):
        # возвращает датафрейм pandas
        if language_name is None:
            return self.raw
        return self.data[self.languages[language_name]]

    def save(self, language_name=None):
        if language_name is None:
            length = int(len(self.raw)/2)
            self.raw[:length].to_csv('source1.tsv', sep="\t")
            self.raw[length:].to_csv('source2.tsv', sep="\t")
        else:
            length = int(len(self.raw) / 3)
            self.data[self.languages[language_name]][:length].to_csv('data_for_pretrain_'+language_name+'1'+'.tsv', sep="\t")
            self.data[self.languages[language_name]][length:length*2].to_csv('data_for_pretrain_'+language_name+'2'+'.tsv', sep="\t")
            self.data[self.languages[language_name]][2*length:].to_csv('data_for_pretrain_'+language_name+'3'+'.tsv', sep="\t")

    def upload(self,path,language_name=None):
        for i in path:
            if language_name is None:
                df1 = pd.read_csv(i,sep = '\t',index_col=[0])
                self.raw = pd.concat([self.raw, df1], ignore_index=True)
            else:
                df1 = pd.read_csv(i, sep='\t', index_col=[0])
                self.data[self.languages[language_name]] = pd.concat([self.data[self.languages[language_name]], df1], ignore_index=True)


    def sentence_uploading(self,language_name,id_and_sent):
        data = self.dict_fill(id_and_sent)
        df1 = pd.DataFrame(columns=self.raw.columns, data=data, index=[0])
        self.data[self.languages[language_name]] = \
            pd.concat([self.data[self.languages[language_name]], df1], ignore_index=True)


