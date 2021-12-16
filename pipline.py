#project = ['accumulo','ambari','beam','cloudstack','flink','incubator-pinot','lucene-solr']
#project = ['beam']
project = ['commons-lang','kafka','shardingsphere']

import pandas as pd
import javalang
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from gensim.models.word2vec import Word2Vec
import multiprocessing
import os,sys,warnings
warnings.filterwarnings('ignore')
sys.setrecursionlimit(50000)

for p in project:
    ratio = '3:1:1'
    print('procesing :',p)
    path = 'data/'+p+'/'+p+'.pkl'
    s = pd.read_pickle(path)
    s.columns = ['old', 'new', 'label']
    #s = s[:5000]
    #get ast
    def parse_program(func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree
    #s['old'] = s['old'].apply(parse_program)
    #s['new'] = s['new'].apply(parse_program)
    #ast_path = 'data/'+p+'/'+p+'_ast.pkl'
    #s.to_pickle(ast_path)
    #get ast seq
    from tree import get_sequence as func
    def trans_to_sequences(cd):
        ast = parse_program(cd)
        sequence = []
        father = []
        func(ast, sequence, father, 0, cd)
        return sequence, father
    s[['old','go']] = s['old'].apply(trans_to_sequences).apply(pd.Series)
    s[['new','gn']] = s['new'].apply(trans_to_sequences).apply(pd.Series)
    s.reset_index(drop=True, inplace=True)
    dell = []
    for i in range(len(s)):
        if len(s['old'][i])>1000 or len(s['new'][i])>1000 or s['old'][i]==s['new'][i]:dell.append(i)
    s.drop(dell,inplace=True)
    s.reset_index(drop=True, inplace=True)
    #make graph
    import numpy as np
    def graph(connection):
        lenx = len(connection)
        tmp = np.zeros((1001,1001),dtype='bool_')
        for i in range(lenx):
            tmp[i][i]=True
            tmp[i][connection[i]] = tmp[connection[i]][i] = True
        return tmp
    s['go'] = s['go'].apply(graph)
    s['gn'] = s['gn'].apply(graph)
    corpus = s['old'] + s['new']
    #word2vec
    w2v = Word2Vec(corpus, size=128, sg=1, window=10 ,min_count = 3, workers=multiprocessing.cpu_count()) # max_final_vocab=3000 tmp ignore
    w2v.save('data/'+p+'/node_w2v_128')

    word2vec = w2v.wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    def to_index(tmp):
        result = []
        for k in tmp:
            result.append(vocab[k].index if k in vocab else max_token)
        while len(result)<=1000:result.append(max_token)
        return result
    s['old'] = s['old'].apply(to_index)
    s['new'] = s['new'].apply(to_index)
    data_num = len(s)
    ratios = [int(r) for r in ratio.split(':')]
    train_split = int(ratios[0]/sum(ratios)*data_num)
    val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

    s = s.sample(frac=1, random_state=666)
    train = s.iloc[:train_split]
    train.reset_index(drop=True, inplace=True)
    dev = s.iloc[train_split:val_split]
    dev.reset_index(drop=True, inplace=True)
    test = s.iloc[val_split:]
    test.reset_index(drop=True, inplace=True)
    s.reset_index(drop=True, inplace=True)
    del s
    trainp = 'data/'+p+'/train.pkl'
    testp = 'data/'+p+'/test.pkl'
    devp = 'data/'+p+'/dev.pkl'
    #pp = 'data/'+p+'/token.pkl'
    #s.to_pickle(pp)
    print('save train')
    train.to_pickle(trainp)
    print('save test')
    test.to_pickle(testp)
    print('save dev')
    dev.to_pickle(devp)