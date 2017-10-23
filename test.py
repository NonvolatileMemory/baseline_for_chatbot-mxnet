from data_utils import *
from data_loader import padding
import mxnet.ndarray as nd
import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon.rnn import LSTM,GRU,LSTMCell,GRUCell
from mxnet import autograd
from mxnet.gluon.nn import Embedding
import pickle
#step_1:load the data
#step_1.1:padding
#step_2:gen_vocab,sen2id
fenci_right_save_path = "/root/PycharmProjects/sigir/data/right_fenci_profile_pad.csv"
fenci_wrong_save_path = "/root/PycharmProjects/sigir/data/wrong_fenci_profile_pad.csv"


single_lstm_model_path = "/root/PycharmProjects/sigir/baseline/dual_lstm/model/single_lstm_model_new.params135"

with open("/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/word2id_file",'rb') as f:
    word2id = pickle.load(f)
with open("/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/id2word_file",'rb') as ff:
    id2word = pickle.load(ff)
vocab_size = len(id2word)

net = gluon.nn.Sequential()

with net.name_scope():
    net.add(Embedding(input_dim=vocab_size,output_dim=1000))
    net.add(LSTM(hidden_size=1000,layout='NTC',num_layers=2))
    net.add(gluon.nn.Dense(2,flatten=True))
net.initialize()


net.load_params(single_lstm_model_path,mx.cpu())
post = "你 喜欢 VR 吗"
respnose = "喜欢"
fake_resp = ['我','喜欢','篮球']
fake_reps = ['我','没有','爸爸']

exp1 = padding(post,17)+str(" ")+padding(respnose,17)
print(exp1)
data = [word2id[s] for s in exp1.split(" ")]
print(data)
data = nd.array(data)
data = data.reshape((1,-1))
for _ in range(1):
    output = net(nd.array(data))
    print(output)