import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random
import time

class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.node_list = []
        self.th = torch.cuda
        self.batch_node = None
        self.max_index = vocab_size
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        return tensor.cuda()

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.embedding_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            # else:
            #     batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramCC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, batch_size, pretrained_weight=None):
        super(BatchProgramCC, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, pretrained_weight)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # hidden
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(in_features = self.hidden_dim * 4,out_features = 2)
        self.dropout = nn.Dropout(0.2)
        #atten
        self.l_w = torch.normal(0,1,size =(self.hidden_dim * 2, self.hidden_dim * 2)).cuda()
        self.l_b = torch.zeros(self.hidden_dim * 2).cuda()
        self.r_w = torch.normal(0,1,size =(self.hidden_dim * 2, self.hidden_dim * 2)).cuda()
        self.r_b = torch.zeros(self.hidden_dim * 2).cuda()
        self.attn_old = nn.TransformerEncoderLayer(d_model = 200, nhead = 8)
        self.attn_new = nn.TransformerEncoderLayer(d_model = 200, nhead = 8)
        self.fc1 = nn.Linear(in_features = 400,out_features = 128)
        self.fc2 = nn.Linear(in_features = 128,out_features = 2)
        self.fc3 = nn.Linear(in_features = 400,out_features = 2)


    def init_hidden(self):
        if isinstance(self.bigru, nn.LSTM):
            h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
            return h0, c0
        return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        return zeros.cuda()

    def encode(self, x):
        ta = time.time()
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        # return encodes
        tb=time.time()
        gru_out, hidden = self.bigru(encodes, self.hidden)
        gru_out = torch.transpose(gru_out, 1, 2)
        #print('gru_out shape : ',gru_out.shape)
        # pooling
        #gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)  #move to astnn
        #gru_out = gru_out[:,-1]

        return gru_out,tb-ta

    def forward(self, x1, x2):
        lvec,tl = self.encode(x1)  #shape [batch,200,len]
        rvec,tr = self.encode(x2)
        #print(lvec.shape,rvec.shape)
        '''l = F.max_pool1d(lvec, lvec.size(2)) #shape bs,200,1
        r = F.max_pool1d(rvec, rvec.size(2))
        lv = torch.cat((lvec,l),2)
        rv = torch.cat((rvec,r),2) 
        lo = self.attn_old(lv.permute(2,0,1)).permute(1,2,0) # shape sl,bs,em
        ro = self.attn_old(rv.permute(2,0,1)).permute(1,2,0)
        lg = F.max_pool1d(lo,lo.size(2)).squeeze(2)
        rg = F.max_pool1d(ro,ro.size(2)).squeeze(2)
        lr = torch.cat((lg,rg),1)
        ot = self.fc1(lr)
        ot = self.fc2(ot)
        ot = F.softmax(ot,dim=1)'''
        #manual attention
        '''
        l_avg = F.max_pool1d(lvec,lvec.size(2)).squeeze(2)
        r_avg = F.max_pool1d(rvec,rvec.size(2)).squeeze(2)
        lvec = lvec.permute(0,2,1)
        rvec = rvec.permute(0,2,1)
        #l_avg = torch.mean(lvec,1) #batch,200
        #r_avg = torch.mean(rvec,1)
        l_att = F.softmax(torch.tanh(torch.einsum('ijk,kl,ilm->ijm',lvec,self.l_w,torch.unsqueeze(r_avg,-1))+self.l_b),dim=1)
        r_att = F.softmax(torch.tanh(torch.einsum('ijk,kl,ilm->ijm',rvec,self.r_w,torch.unsqueeze(l_avg,-1))+self.r_b),dim=1)
        l_req = torch.sum(l_att*lvec,1)
        r_req = torch.sum(r_att*rvec,1)
        req = torch.cat((l_req,r_req),1)
        ot = self.fc3(req)
        #ot = self.fc2(ot)
        ot = F.softmax(ot,dim = 1)
        #'''
        #astnn
        #'''
        #print(lvec.shape)
        lvec = F.max_pool1d(lvec, lvec.size(2)).squeeze(2)
        rvec = F.max_pool1d(rvec, rvec.size(2)).squeeze(2)
        #abs_dist = torch.abs(torch.add(lvec, -rvec))
        abs_dist = torch.cat([lvec,rvec],1)
        #print(abs_dist.shape)
        #y = torch.sigmoid(self.hidden2label(abs_dist))
        y = self.fc(abs_dist)
        #print(y.shape)
        ot = F.softmax(y,dim = 1)
        #'''
        return ot,tl,tr

