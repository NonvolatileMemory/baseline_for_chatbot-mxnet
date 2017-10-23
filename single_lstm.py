from data_utils import *
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


single_lstm_model_path = "/root/PycharmProjects/sigir/baseline/dual_lstm/model/single_lstm_model.params57"


raw_data,raw_label = load_data_and_labels(fenci_right_save_path,fenci_wrong_save_path)


with open('/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/word2id_file', 'rb') as f:
    word2id = pickle.load(f)
with open('/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/id2word_file','rb') as ff:
    id2word = pickle.load(ff)

vocab_size = len(id2word)
data,label = build_input_data(sentences=raw_data,labels=raw_label,vocabulary=word2id)
train_dataset = gluon.data.ArrayDataset(data[0:9000], label[0:9000])
data_iter = gluon.data.DataLoader(train_dataset, 512, shuffle=True)


net = gluon.nn.Sequential()

with net.name_scope():
    net.add(Embedding(input_dim=vocab_size,output_dim=1000))
    net.add(LSTM(hidden_size=1000,layout='NTC',num_layers=2,bidirectional = False))
    net.add(gluon.nn.Dense(2,flatten=True))
net.initialize()


trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.05})

epochs = 10000
batch_size = 512

#no use for new embedding api
#def get_input_data(data,vocab_size):
#    return [nd.one_hot(X,vocab_size).asnumpy() for X in data]


net.load_params(single_lstm_model_path,mx.cpu())

for e in range(epochs):
    total_loss = 0
    count = 0

    for train_data,train_label in data_iter:
        count = count + 1
        with autograd.record():
            output = net(train_data)
            loss =  gluon.loss.SoftmaxCrossEntropyLoss()(output,train_label)
        loss.backward()
        trainer.step(batch_size)
        batch_loss = nd.sum(loss)
        print("Epoch %d, count %d, loss: %f " % (e,count,batch_loss.asscalar()))

    #ACC
    acc = mx.metric.Accuracy()
    test_output = net(nd.array(data[9001:-1]))
    acc.update(preds=[test_output], labels=[nd.array(label[9001:-1])])
    print(acc.get())
    #LOSS
    total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/5000))
    #SAVE
    single_lstm_model_path = "/root/PycharmProjects/sigir/baseline/dual_lstm/model/single_lstm_model_new.params" + str(e)
    f = open(single_lstm_model_path, 'w')
    f.close()
    net.save_params(single_lstm_model_path)