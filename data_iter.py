from torch.utils.data import Dataset, DataLoader, Sampler
import random,math
import pandas as pd
import numpy as np


random.seed(20)

#Dataset
class MyClassBalanceDataset(Dataset):
    def __init__(self, root):
        super(MyClassBalanceDataset, self).__init__()
        labels = []
        old = []
        new = []
        go = []
        gn = []
        source = pd.read_pickle(root)
        
        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['go'] = source['go'].apply(graph)
        source['gn'] = source['gn'].apply(graph)

        self.len = len(source)
        self.label = source['label'].tolist()
        self.old = source['old'].tolist()
        self.new = source['new'].tolist()
        self.go = source['go'].tolist()
        self.gn = source['gn'].tolist()
        self.one = []
        self.zero = []
        for i in range(len(self.label)):
            if self.label[i]==1:self.one.append(i)
            else :self.zero.append(i)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.old[index],self.go[index],self.new[index],self.gn[index],self.label[index]

class MyBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, class_weight):
        super(MyBatchSampler, self).__init__(data_source)
        random.seed(20)
        self.data_source = data_source
        assert isinstance(class_weight, list)
        assert 1 - sum(class_weight) < 1e-5
        self.batch_size = batch_size

        _num = len(class_weight)
        number_in_batch = {i: 0 for i in range(_num)}
        for c in range(_num):
            number_in_batch[c] = math.floor(batch_size * class_weight[c])
        _remain_num = batch_size - sum(number_in_batch.values())
        number_in_batch[random.choice(range(_num))] += _remain_num
        self.number_in_batch = number_in_batch
        self.offset_per_class = {i: 0 for i in range(_num)}
        #print(f'setting number_in_batch: {number_in_batch}')
        #print('my sampler is inited.')

    def __iter__(self):
        #print('======= start __iter__ =======')
        batch = []
        i = 0
        while i < len(self):
            for c, num in self.number_in_batch.items():
                start = 0
                end = 0
                if c==0:
                    end = len(self.data_source.zero)
                    for _ in range(num):
                        idx = start + self.offset_per_class[c]
                        if idx >= end:
                            self.offset_per_class[c] = 0
                        idx = start + self.offset_per_class[c]
                        batch.append(self.data_source.zero[idx])
                        #batch.append(0)
                        self.offset_per_class[c] += 1
                else: 
                    end = len(self.data_source.one)
                    for _ in range(num):
                        idx = start + self.offset_per_class[c]
                        if idx >= end:
                            self.offset_per_class[c] = 0
                        idx = start + self.offset_per_class[c]
                        batch.append(self.data_source.one[idx])
                        #batch.append(0)
                        self.offset_per_class[c] += 1

            assert len(batch) == self.batch_size
            # random.shuffle(batch)
            yield batch
            batch = []
            i += 1

    def __len__(self):
        return len(self.data_source) // self.batch_size



class MyDataset(Dataset):
    def __init__(self,file_path):
        labels = []
        old = []
        new = []
        go = []
        gn = []
        source = pd.read_pickle(file_path)

        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['go'] = source['go'].apply(graph)
        source['gn'] = source['gn'].apply(graph)

        self.len = len(source)
        self.label = source['label'].tolist()
        self.old = source['old'].tolist()
        self.new = source['new'].tolist()
        self.go = source['go'].tolist()
        self.gn = source['gn'].tolist()

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.old[index],self.go[index],self.new[index],self.gn[index],self.label[index]

