import time
import pandas as pd
import numpy as np
import re
import string

# print(re.sub(r'\[\d{1,}\]',"",t))

#
df1 = pd.read_csv('patents_ru_and_en/data_ru1.csv')

df2 = pd.read_csv('patents_ru_and_en/data_en1.csv')

# df1.to_csv('patents_ru_and_en/data_ru1.csv',index=False)
# df2.to_csv('patents_ru_and_en/data_en1.csv',index=False)


#Non-patent Document [Патентные документы] Патентная публикация Кореи
#(MOL) (CDX)
'''
    Чистим данные от таблиц руками
'''
# arr = []
# for i in range(len(df2)):
#     if 'TABLE' in df2.values[:,2][i]:
#         arr.append([i,df2.values[:,0][i]])
#
# print(arr)
# print(df1.values[:,2][arr[0][0]])
# print(df2.values[:,2][arr[0][0]])
#
# t1 = df1.values[:,2][arr[0][0]]
# t2 = df2.values[:,2][arr[0][0]]
#
#
# restriction1_zh = re.search(r'编号 可溶性融合蛋白', t1)
# restriction2_zh = re.search(r'本发明还涉及一种核酸', t1)
# print(t1[restriction1_zh.span()[0]:restriction2_zh.span()[1]])


# restriction1_en = re.search(r'TABLE-US-00001', t2)
# restriction2_en = re.search(r'  The present invention also relates to a nucleic acid', t2)

# res = re.search(r'In one or more implementation modes:',t2)

#BRIEF DESCRIPTION OF THE
# new_data_zh = t1[:restriction1_zh.span()[0]] + t1[restriction2_zh.span()[0]:]

# new_data_en = t2[:restriction1_en.span()[0]-1] + t2[restriction2_en.span()[0]:]


# print(new_data_zh)
# print(new_data_en)
#
#
# df1.at[arr[0][0],'abstr_p2']=new_data_zh
# df2.at[arr[0][0],'abstr_p2']=new_data_en
#
# df1.to_csv('clear_data_zh',index=False)
# df2.to_csv('clear_data_en',index=False)

'''
    Дополнительная коррекция 
'''
#
# arr = []
# for i in range(len(df2)):
#     if ' DRAWING\n' in df2.values[:,2][i]:
#         arr.append([i,df2.values[:,0][i]])
#
# print(arr)
# print(len(arr))
#
# for i in arr:
#     text = df2.values[:,2][i[0]]
#     res = re.search(r' DRAWING\n',text)
#     new_text = text[:res.span()[1]]
#     df2.at[i[0], 'abstr_p2'] = new_text
#     print(df2.values[:,2][i[0]])
#
# df2.to_csv('clear_data_en',index=False)
