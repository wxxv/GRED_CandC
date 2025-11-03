import re
import pickle
import json
import numpy as np
import pandas as pd
import nltk
from functools import reduce
from collections import Counter
from src.utils import data_loader_semQL



class Vocab(object):
    def __init__(self, path_in, path_out = None):
        # vocab for input
        self.word2idx, self.idx2word = self.get_vocab(path_in)
        self.size = len(self.word2idx)
        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        
        # vocab for output
        if path_out is not None:
            self.word2idx_out, self.idx2word_out = self.get_vocab(path_out)
            self.size_out = len(self.word2idx_out)
            # self.word2idx_out, self.idx2word_out = check_emb_vocab(path_out)
            # self.size_out = len(self.word2idx_out)
        else:
            self.word2idx_out, self.idx2word_out, self.size_out = self.word2idx, self.idx2word, self.size


    def get_emb_vocab(selg, path):
        emb=pickle.load(open(path, 'rb'))


    def get_vocab(self, path):
        word2idx = {}
        idx2word = []
        with open(path) as f:
            for line in f:
                w = line.split()[0]
                word2idx[w] = len(word2idx)
                idx2word.append(w)
        return word2idx, idx2word


    def merge(self, vocab_merged):
        words_merged = list(vocab_merged.word2idx.keys())
        unused_index = 0
        for word in words_merged:
            if word not in self.word2idx:
                index = self.word2idx.pop('[unused{}]'.format(unused_index))
                self.word2idx[word] = index
                self.idx2word[index] = word
                unused_index += 1

    @staticmethod
    def build(sents, path, size = None):
        v = ['<pad>', '<go>', '<eos>', '<unk>']
        words = [w for s in sents for w in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            if w.strip() == '':
                continue
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))
        print('save {} words in vocab file {}'.format(len(v), path))

    def sen2indexes(self, sentences):
        '''sentence: a string or list of string
           return: a numpy array of word indices
        '''
        def convert_sent(sent, maxlen):
            idxes = np.zeros(maxlen, dtype=np.int64)
            idxes.fill(self.pad)
            tokens = nltk.word_tokenize(sent.strip())
            idx_len = min(len(tokens), maxlen)
            for i in range(idx_len): idxes[i] = self.word2idx.get(tokens[i], self.unk)
            return idxes, idx_len

        if type(sentences) is list:
            inds, lens = [], []
            for sent in sentences:
                idxes, idx_len = convert_sent(sent, maxlen)
                inds.append(idxes)
                lens.append(idx_len)
            return np.vstack(inds), np.vstack(lens)
        else:
            inds, lens = self.sent2indexes([sentences], maxlen)
            return inds[0], lens[0]


def build_vocab(data_path, table_path, input_vocab_file, output_vocab_file): 
    with open(data_path, 'r') as f:
        data = json.load(f)
    tables = data_loader_semQL.get_tables(table_path)
    examples = [data_loader_semQL.to_example(x, tables[x['db_id']]) for x in data]
    examples = list(filter(lambda x : x is not None, examples))
    
    # input vocab
    #print(examples[0].src_sent)
    texts = [example.text_seqs +  reduce(lambda x, y: x +y, example.src_sent) + example.node_seqs + reduce(lambda x, y: x + y, example.schema_seqs) + example.sql_seqs for example in examples]
    Vocab.build(texts, input_vocab_file)

    ## output vocab
    #texts = [reduce(lambda x, y: x + y, example.schema_seqs) + example.sql_seqs for example in examples]
    #Vocab.build(texts, output_vocab_file)

def check_emb_vocab(filename):
    data=pickle.load(open(filename,'rb'))
    print(data)
    exit

def load_word_emb(file_name, use_small=False):
    print ('Loading word embedding from %s'%file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x:float(x), info[1:])))
    if '<unk>' not in ret:
        ret['<unk>'] = np.random.rand(300)
    return ret

def get_vocab_glove_embeddding(glove_file, vocab_file, embedding_path):
    vocab = Vocab(vocab_file)
    glove_emb = load_word_emb(glove_file, False)

    word2idx = vocab.word2idx
    vocab_emb = []
    num_unk = 0
    for word, idx in word2idx.items():
        if word in glove_emb:
            vocab_emb.append(glove_emb[word])
        else:
            print(word)
            num_unk += 1
            vocab_emb.append(glove_emb['<unk>'])
            #vocab_emb.append(np.random.rand(300))

    vocab_emb = np.array(vocab_emb, dtype = np.float32)
    print(vocab_emb.shape, num_unk)
    with open(embedding_path, 'wb') as f:
        pickle.dump(vocab_emb, f)
        print('vocab emb of size {} is saved in {}'.format(len(vocab_emb), embedding_path))
