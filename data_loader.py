import pandas as pd
import jieba
import csv
import numpy as np
import random
from data_utils import build_vocab
#this file is used to be a negtive label generator
data_path = "/root/PycharmProjects/sigir/data/profile.csv"
fenci_right_save_path = "/root/PycharmProjects/sigir/data/right_fenci_profile_pad.csv"
fenci_wrong_save_path = "/root/PycharmProjects/sigir/data/wrong_fenci_profile_pad.csv"

def padding(sentence,padding_len):
    len_zero = padding_len - len(sentence.split(" "))
    for _ in range(len_zero):
        sentence = sentence + str(" ")
        sentence = sentence+ str("PAD")
    return sentence

df = pd.read_csv(data_path)

Post = []
Response = []

for sentence in df['Post']:
    seg_list = jieba.cut(sentence)
    sentence = " ".join(seg_list)
    Post.append(sentence)

for sentence in df['Response']:
    seg_list = jieba.cut(sentence)
    sentence = " ".join(seg_list)
    Response.append(sentence)
Post = np.array(Post)
Response = np.array(Response)
All = np.concatenate((Post.reshape((-1,1)), Response.reshape(-1,1)), axis=1)
with open(fenci_right_save_path,'w') as f_r:
    with open(fenci_wrong_save_path,'w') as f_w:
        count = 0
        for row in All:

            f_r.write(padding(row[0],17))
            f_r.write(' ')
            f_r.write(padding(row[1],17))
            f_r.write('\n')

            rand = random.randint(10,2000)
            f_w.write(padding(row[0],17))
            f_w.write(' ')
            f_w.write(padding(All[(rand+count)%5000][1],17))
            f_w.write('\n')

            count+= count
#print(Post[900])
#print(list(All))