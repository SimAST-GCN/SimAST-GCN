#project = ['accumulo','ambari','beam','cloudstack','flink','incubator-pinot','lucene-solr']
project = ['accumulo','ambari','beam','cloudstack','commons-lang','flink','incubator-pinot','kafka','lucene-solr','shardingsphere']
#project = ['shardingsphere']

import pandas as pd
from gensim.models.word2vec import Word2Vec
import multiprocessing
import os

def transform(tmp):
    tmp = tmp.replace("'",' ').replace(':',' ').replace('"',' ').replace('\n',' ').replace('?',' ').replace(';',' ').replace('(',' ').replace('.',' ').r
eplace(')',' ').replace('{',' ').replace('}',' ').replace('=',' ').replace('[',' ').replace(']',' ').replace('<',' ').replace('>',' ').replace(',',' ').
replace('+',' ').replace('-',' ').replace('/',' ').replace('*',' ').replace('!',' ')
    tmp = tmp.split()
    #tmp = ' '.join(tmp)
    return tmp

def tf(tmp):
    tmp = tmp.replace("'",' ').replace(':',' ').replace('"',' ').replace('\n',' $ ').replace('?',' ').replace(';',' ').replace('(',' ').replace('.',' ')
.replace(')',' ').replace('{',' ').replace('}',' ').replace('=',' ').replace('[',' ').replace(']',' ').replace('<',' ').replace('>',' ').replace(',',' '
).replace('+',' ').replace('-',' ').replace('/',' ').replace('*',' ').replace('!',' ')
    tmp = tmp.split()
    tmp = ' '.join(tmp)
    ever_alp = 0
    is_sp = 0
    tmp = list(tmp)
    for i in range(len(tmp)):
        if (tmp[i]>='a' and tmp[i]<='z') or (tmp[i]>='A' and tmp[i]<='Z'):
            ever_alp = 1
            is_sp = 0
        if tmp[i]=='$' and ever_alp==0:tmp[i]=' '
        if ever_alp==1:
            if tmp[i]=='$' and is_sp==0:is_sp = 1
            elif tmp[i]=='$' and is_sp==1:tmp[i]=' '
    if tmp[-1]=='$':tmp[-1]=' '
    tmp = ''.join(tmp) 
    tmp = tmp.split()
    tmp = ' '.join(tmp)
    tmp = tmp.replace(' $ ','$')
    while not ( (tmp[-1]>='a' and tmp[-1]<='z') or (tmp[-1]>='A' and tmp[-1]<='Z') ):
        tmp = tmp[:-1]
    return tmp



for p in project:
    ratio = '3:1:1'
    print('procesing :',p)
    path = 'data/'+p+'/'+p+'.pkl'
    s = pd.read_pickle(path)
    s.columns = ['old', 'new', 'label']
    corpus = s['old'].apply(transform)
    corpus += s['new'].apply(transform)
    w2v = Word2Vec(corpus, size=128, sg=1, window=5 ,min_count = 3, workers=multiprocessing.cpu_count()) # max_final_vocab=3000 tmp ignore
    if not os.path.exists('data/'+p+'/token'):
        os.mkdir('data/'+p+'/token')
    w2v.save('data/'+p+'/token/node_w2v_128')

    word2vec = w2v.wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    def to_index(tmp):
        tmp = transform(tmp)
        idx = []
        for _ in tmp:
            idx.append(vocab[_].index if _ in vocab else max_token)
        while len(idx)<5000:idx.append(max_token)
        return idx
    
    s['old'] = s['old'].apply(to_index)
    s['new'] = s['new'].apply(to_index)

    data_num = len(s)
    ratios = [int(r) for r in ratio.split(':')]
    train_split = int(ratios[0]/sum(ratios)*data_num)
    val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

    s = s.sample(frac=1, random_state=666)
    train = s.iloc[:train_split]
    dev = s.iloc[train_split:val_split]
    test = s.iloc[val_split:]

    trainp = 'data/'+p+'/token/'+p+'_token_train.pkl'
    testp = 'data/'+p+'/token/'+p+'_token_test.pkl'
    devp = 'data/'+p+'/token/'+p+'_token_dev.pkl'
    pp = 'data/'+p+'/token/'+p+'_token.pkl'
    s.to_pickle(pp)
    train.to_pickle(trainp)
    test.to_pickle(testp)
    dev.to_pickle(devp)
