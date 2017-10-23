from data_utils import *
import pickle

fenci_right_save_path = "/root/PycharmProjects/sigir/data/right_fenci_profile_pad.csv"
fenci_wrong_save_path = "/root/PycharmProjects/sigir/data/wrong_fenci_profile_pad.csv"

raw_data, raw_label = load_data_and_labels(fenci_right_save_path, fenci_wrong_save_path)
word2id, id2word = build_vocab(raw_data)

with open('/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/word2id_file', 'wb') as f:
    pickle.dump(word2id, f)
    f.close()
with open('/root/PycharmProjects/sigir/baseline/dual_lstm/vocab/id2word_file', 'wb') as ff:
    pickle.dump(id2word, ff)
    ff.close()