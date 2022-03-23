from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests

model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ru')

#url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=19008416,18927361,18787170&rettype=txt&retmode=text'
#r = requests.get(url)
#r.text
#soup = BeautifulSoup(r.text)

text = 'The enzyme mTOR (mammalian target of rapamycin) is a major target for therapeutic \
intervention to treat many human diseases, including cancer, but very little is \
known about the processes that control levels of mTOR protein. Here, we show that \
mTOR is targeted for ubiquitination and consequent degradation by binding to the \
tumor suppressor protein FBXW7. Human breast cancer cell lines and primary tumors \
showed a reciprocal relation between loss of FBXW7 and deletion or mutation of \
PTEN (phosphatase and tensin homolog), which also activates mTOR. Tumor cell \
lines harboring deletions or mutations in FBXW7 are particularly sensitive to \
rapamycin treatment, which suggests that loss of FBXW7 may be a biomarker for \
human cancers susceptible to treatment with inhibitors of the mTOR pathway.'

batch = tokenizer([text], return_tensors="pt")
generated_ids = model.generate(**batch)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
