#project = ['accumulo','ambari','cloudstack','flink','incubator-pinot','lucene-solr','commons-lang','kafka','shardingsphere']
#project = ['ambar','bea']
project = ['flink','cloudstack']

import pandas as pd
import javalang
import warnings
warnings.filterwarnings('ignore')
from gensim.models.word2vec import Word2Vec
import multiprocessing
import os,sys,warnings
warnings.filterwarnings('ignore')
sys.setrecursionlimit(50000)
import dask.dataframe as dd
from dask.distributed import Client
import os
from multiprocessing import freeze_support
from dask.distributed import progress
import dask.multiprocessing
import swifter
import numpy as np


if __name__=="__main__":
    #freeze_support()
    #client=Client(n_workers=10, threads_per_worker=2)

    for p in project:
        ratio = '3:1:1'
        path = 'data/'+p+'/'+p+'.pkl'
        s = pd.read_pickle(path)
        print('procesing :',p,", total_size :",len(s))
        #s = dd.from_pandas(s, chunksize=1000)
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
            if len(sequence)>=1000:
                sequence=sequence[:1000]
                father = father[:1000]
            return sequence, np.array(father,dtype='int32')
        s[['old','go']] = s['old'].swifter.allow_dask_on_strings(enable=True).apply(trans_to_sequences).swifter.apply(pd.Series)
        s[['new','gn']] = s['new'].swifter.allow_dask_on_strings(enable=True).apply(trans_to_sequences).swifter.apply(pd.Series)
        #s["old_temp"] = s['old'].apply(trans_to_sequences,meta=("old_temp",object))
        #s["old"] = s["old_temp"].apply(lambda x:x[0],meta=("old",object))
        #s["go"] = s["old_temp"].apply(lambda x:x[1],meta=("go",object))

        #s["new_temp"] = s['new'].apply(trans_to_sequences,meta=("new_temp",object))
        #s["new"] = s["new_temp"].apply(lambda x:x[0],meta=("new",object))
        #s["gn"] = s["new_temp"].apply(lambda x:x[1],meta=("gn",object))

        s.reset_index(drop=True, inplace=True)
        dell = []
        for i in range(len(s)):
            if s['old'][i]==s['new'][i]:dell.append(i)
        print("delte_size : ",len(dell))
        s.drop(dell,inplace=True)
        s.reset_index(drop=True, inplace=True)

        #make graph
        '''
        import numpy as np
        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        s['go'] = s['go'].apply(graph)#do not use apply or error
        s['gn'] = s['gn'].apply(graph)
        '''
        corpus = s['old'] + s['new']
        #word2vec
        w2v = Word2Vec(corpus, size=300, sg=1, window=5, min_count = 3,workers=multiprocessing.cpu_count()) # max_final_vocab=3000 tmp ignore
        w2v.save('data/'+p+'/node_w2v_128')

        word2vec = w2v.wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        print("max_token : ",max_token)

        #now consider to remove the part, to get small size
        
        def to_index(tmp):
            result = []
            for k in tmp:
                result.append(vocab[k].index if k in vocab else max_token)
            while len(result)<1000:result.append(max_token)
            return np.array(result,dtype = 'int32')
        s['old'] = s['old'].swifter.apply(to_index)
        s['new'] = s['new'].swifter.apply(to_index)
        
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
        