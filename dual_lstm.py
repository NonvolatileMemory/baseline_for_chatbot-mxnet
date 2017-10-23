import pickle
from data_utils import *
import mxnet.ndarray as nd
import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon.rnn import LSTM,GRU,LSTMCell,GRUCell
from mxnet import autograd
from mxnet.gluon.nn import Embedding

import os
#step_1:load the data
#step_1.1:padding
#step_2:gen_vocab,sen2id
fenci_right_save_path = "/root/PycharmProjects/sigir/data/right_fenci_profile_pad.csv"
fenci_wrong_save_path = "/root/PycharmProjects/sigir/data/wrong_fenci_profile_pad.csv"


single_lstm_model_path = "/root/PycharmProjects/sigir/baseline/dual_lstm/model/dual_lstm_model.params"

#read data
raw_data,raw_label = load_data_and_labels(fenci_right_save_path,fenci_wrong_save_path)
#read vocab
with open("/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/word2id_file",'rb') as f:
    word2id = pickle.load(f)
with open("/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/id2word_file",'rb') as ff:
    id2word = pickle.load(ff)

vocab_size = len(id2word)
#gen batch and val_set
data,label = build_input_data(sentences=raw_data,labels=raw_label,vocabulary=word2id)
train_dataset = gluon.data.ArrayDataset(data[0:9000], label[0:9000])
data_iter = gluon.data.DataLoader(train_dataset, 512, shuffle=True)


post_net = gluon.nn.Sequential()
with post_net.name_scope():
    post_net.add(Embedding(input_dim=vocab_size,output_dim=1000))
    post_net.add(LSTM(hidden_size=1000,layout='NTC',num_layers=2,bidirectional = False))
post_net.initialize()


resp_net = gluon.nn.Sequential()
with resp_net.name_scope():
    resp_net.add(Embedding(input_dim=vocab_size,output_dim=1000))
    resp_net.add(LSTM(hidden_size=1000,layout='NTC',num_layers=2,bidirectional = False))
resp_net.initialize()


mlp_net = gluon.nn.Sequential()
with mlp_net.name_scope():
    mlp_net.add(gluon.nn.Dense(2,flatten=True))
mlp_net.initialize()


trainer = gluon.Trainer(
    post_net.collect_params() and resp_net.collect_params() and mlp_net.collect_params(), 'sgd', {'learning_rate': 0.05})

epochs = 10000
batch_size = 512


def get_input_data(data,vocab_size):
    return [nd.one_hot(X,vocab_size).asnumpy() for X in data]


#net.load_params(single_lstm_model_path,mx.cpu())

for e in range(epochs):
    total_loss = 0
    count = 0

    for train_data,train_label in data_iter:
        count = count + 1
        #print(train_data)
        #print("------------------------post----------------")
        post = (train_data.T[0:17]).T
        #print(post)
        #print("------------------------resp----------------")
        resp = (train_data.T[17:34]).T
        #print(resp)
        with autograd.record():
            post_output = post_net(post)
            #print("-----------------post-out-----------------------")
            #print(post_output)
            resp_output = resp_net(resp)
            #print("-----------------resp-out-----------------------")
            #print(resp_output)
            #print(mx.ndarray.concat(post_output,resp_output,dim=1))
            output = mlp_net(mx.ndarray.concat(post_output,resp_output,dim=1))
            loss =  gluon.loss.SoftmaxCrossEntropyLoss()(output,train_label)
        loss.backward()
        trainer.step(batch_size)
        batch_loss = nd.sum(loss)
        print("Epoch %d, count %d, loss: %f " % (e,count,batch_loss.asscalar()))

    #ACC
    acc = mx.metric.Accuracy()
    test_data = nd.array(data[9001:-1])
    test_post = (test_data.T[0:17]).T
    test_resp = (test_data.T[17:34]).T
    test_post_output = post_net(test_post)
    test_resp_output = resp_net(test_resp)
    test_output = mlp_net(mx.ndarray.concat(test_post_output,test_resp_output,dim=1))
    acc.update(preds=[test_output], labels=[nd.array(label[9001:-1])])
    print(acc.get())
    #LOSS
    total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/5000))
    #SAVE
    dual_lstm_model_path_post = "/root/PycharmProjects/sigir/baseline/dual_lstm/model/dual_lstm_model_post.params" + str(e)
    dual_lstm_model_path_resp = "/root/PycharmProjects/sigir/baseline/dual_lstm/model/dual_lstm_model_resp.params" + str(e)
    dual_lstm_model_path_mlp = "/root/PycharmProjects/sigir/baseline/dual_lstm/model/dual_lstm_model_mlp.params" + str(e)
    f = open(dual_lstm_model_path_post, 'w')
    f.close()
    f = open(dual_lstm_model_path_resp, 'w')
    f.close()
    f = open(dual_lstm_model_path_mlp, 'w')
    f.close()
    post_net.save_params(dual_lstm_model_path_post)
    resp_net.save_params(dual_lstm_model_path_resp)
    mlp_net.save_params(dual_lstm_model_path_mlp)