import sys
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

from gensim.models import Word2Vec

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, aug_type, aug_size, do_rate, hid_dim, result_queue, SEED):
    def sample_():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)
    
    def sample_emb():
        orig_train = dict()
        training_seqs = tuple()
        
        for user in tqdm(range(1, usernum+1), desc='Preparing original sample dict'):
            seq = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            nxt = user_train[user][-1]   ## last item in item seq
            neg_list = np.zeros([aug_size, maxlen], dtype=np.int32)
    
            idx = maxlen - 1
            ts = set(user_train[user])  ## targets..!
            for i in reversed(user_train[user][:-1]):     
                seq[idx] = i
                pos[idx] = nxt             ## pos seq ( 1~n 까지의 item seq )
                if (nxt != 0): 
                    for it in range(aug_size):
                        neg_list[it][idx] = random_neq(1, itemnum + 1, ts) 
                nxt = i
                idx -= 1
                if idx == -1: break
                    
            orig_train[user] = (seq, pos, neg_list)
            str_seq = list(map(lambda x: str(x), seq))
            training_seqs += (str_seq,)
        
        #print('num of training seqs:', len(training_seqs))
        return orig_train, training_seqs

    def get_pre_emb(seq_data, size=hid_dim):  # 1 for skip_gram
        wv_model = Word2Vec(sentences=seq_data, size=size, window=3, min_count=1, workers=4, sg=0)
        vocab = wv_model.wv.index2word
        print('pre-emb vocab size:' , len(vocab))  ## min_count = 2 --> # 3316 / =2 --> # 3390

        return wv_model
    
    def aug_subst(orig_train, wv_model):
        aug_train = dict()
        aug_len, aug_var = 3, 3
        #aug_len = round(maxlen * do_rate)
        
        for uid in tqdm(range(1, usernum+1), desc='Preparing augmented sample dict'):
            seq = orig_train[uid][0]
            pos = orig_train[uid][1]
            neg_list = orig_train[uid][2]
        
            for it in reversed(range(aug_len + 1)):  # it = 0, 1, 2, 3
                n_seq = seq.copy()
                n_pos = pos.copy()
                
                # original sequence 
                if it == 0:
                    n_uid = uid * aug_size
                    n_neg = neg_list[it]
                    aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
                
                else: # 3 * 3 = 9 aug samples!
                    subs_idx = random.sample(range(1, maxlen-1), aug_len)
                    for s_idx in subs_idx:
                        ori_item = str(n_seq[s_idx])
                        for var in range(aug_var):  # var = 0, 1, 2
                            try:
                                sim_item = wv_model.wv.most_similar(ori_item)[var][0]
                                n_seq[s_idx] = int(sim_item)
                                #n_pos[s_idx-1] = int(sim_item)
                            except KeyError:
                                pass
                            
                            n_it = it * aug_len - var
                            n_uid = uid * aug_size - n_it
                            n_neg = neg_list[n_it]
                            aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
        
        print('aug len & aug size & seq drop rate : ', aug_len, '/ ', aug_size, '/ ', do_rate) 
        print('# of augmented trainining samples: ', len(aug_train)) 
        return aug_train
         
    def sample_slide():
        orig_train = dict()
        
        for user in tqdm(range(1, usernum+1), desc='Preparing original sample dict'):
            seq = np.zeros([maxlen+aug_size-1], dtype=np.int32)  ## 41(maxlen) + 10 - 1 = 50
            pos = np.zeros([maxlen+aug_size-1], dtype=np.int32)
            nxt = user_train[user][-1]   ## last item in item seq
            #neg_list = np.zeros([aug_size, maxlen], dtype=np.int32)
    
            idx = maxlen + aug_size - 2 ## idx = 49 > 48 > ... > 0
            #ts = set(user_train[user])  ## targets..!
            for i in reversed(user_train[user][:-1]): 
                seq[idx] = i
                pos[idx] = nxt             ## pos seq ( item seq of 1~n )
                nxt = i
                idx -= 1
                if idx == -1: break

            orig_train[user] = (seq, pos)
        
        return orig_train
    
    def aug_slide(orig_train):
        aug_train = dict()
        do_cnt = round(maxlen * do_rate)
        n_uid = 0
        
        for uid in tqdm(range(1, usernum+1), desc='Preparing augmented sample dict'):
            seq = orig_train[uid][0]
            pos = orig_train[uid][1]
            neg = np.zeros([maxlen], dtype=np.int32)
            ts = set(user_train[uid])  ## targets..!
        
            for it in reversed(range(aug_size)):  ## it = 9, 8, 7, .., 0
                #n_uid = uid * aug_size - it       ## 1 * 10 - 9 = 1
                
                if it == (aug_size - 1):
                    n_seq = seq.copy()[it:]
                    n_pos = pos.copy()[it:]
                
                else:
                    n_seq = seq.copy()[it: maxlen+it]
                    n_pos = pos.copy()[it: maxlen+it]
                
                    if sum(n_seq) == 0:
                        break
                    
                n_neg = neg.copy()
                for idx in range(maxlen): ## 0,1,2,..,40
                    tmp = n_seq[-(idx+1)]        
                    if (tmp != 0): 
                        n_neg[-(idx+1)] = random_neq(1, itemnum + 1, ts)
                    else:
                        break
                        
                ## drop_out add
                #drop_idx = random.sample(range(1, maxlen-1), do_cnt)
                #for d_idx in drop_idx:
                #    n_seq[d_idx] = 0  
                        
                n_uid += 1
                aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
        
        print('aug size & max_len & do_cnt: ', aug_size, '/ ', maxlen, '/', do_cnt) 
        print('# of augmented trainining samples: ', len(aug_train), n_uid) 
        return aug_train
   
    def sample_subset():
        orig_train = dict()
        
        for user in tqdm(range(1, usernum+1), desc='Preparing original sample dict'):
            seq = np.zeros([maxlen], dtype=np.int32)  ## 41(maxlen) + 10 - 1 = 50
            pos = np.zeros([maxlen], dtype=np.int32)
            nxt = user_train[user][-1]   ## last item in item seq
            #neg_list = np.zeros([aug_size, maxlen], dtype=np.int32)
    
            idx = maxlen - 1 ## idx = 49 > 48 > ... > 0
            #ts = set(user_train[user])  ## targets..!
            for i in reversed(user_train[user][:-1]): 
                seq[idx] = i
                pos[idx] = nxt             ## pos seq
                nxt = i
                idx -= 1
                if idx == -1: break

            orig_train[user] = (seq, pos)
        
        return orig_train
    
    def aug_subset(orig_train):
        aug_train = dict()
        do_cnt = round(maxlen * do_rate)
        n_uid = 0
        
        for uid in range(1, usernum+1):
            seq = orig_train[uid][0]
            pos = orig_train[uid][1]
            neg = np.zeros([maxlen], dtype=np.int32)
            ts = set(user_train[uid])  ## targets..!
            #neg_list = orig_train[uid][2]
        
            if sum(seq) == 0:
                continue
                
            aug_cnt = 0  
            for idx in range(maxlen):  ## it = 0, 1, 2, ..
                if seq[idx] != 0:
                    start = idx
                    break
                else:
                    idx += 1
                
            for it in range(maxlen):
                if (it == start) or (it == 0):
                    n_seq = seq.copy()
                    n_pos = pos.copy()
                else:    
                    n_seq = np.concatenate( ( np.zeros(it), seq.copy()[:-it] ), axis=None)
                    n_pos = np.concatenate( ( np.zeros(it), pos.copy()[:-it] ), axis=None)
                
                n_neg = neg.copy()
                for idx in range(maxlen): ## 0,1,2,..,40
                    tmp = n_seq[-(idx+1)]        
                    if (tmp != 0): 
                        n_neg[-(idx+1)] = random_neq(1, itemnum + 1, ts)
                    else:
                        break
                        
                # drop_out add
                drop_idx = random.sample(range(1, maxlen-1), do_cnt)
                for d_idx in drop_idx:
                    n_seq[d_idx] = 0
                    
                if sum(n_seq) == 0:
                    break
                    
                n_uid += 1
                aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
                
                aug_cnt += 1
                if aug_cnt == aug_size:
                    break
        
        print('aug size & max_len : ', aug_size, '/ ', maxlen, '/', do_cnt) 
        print('# of augmented trainining samples: ', len(aug_train)) 
        return aug_train
    
    
    def sample_drop():
        orig_train = dict()
        aug_len = round(maxlen * do_rate) 
        
        for user in tqdm(range(1, usernum+1), desc='Preparing original sample dict'):
            seq = np.zeros([maxlen+aug_len], dtype=np.int32)
            pos = np.zeros([maxlen+aug_len], dtype=np.int32)
            nxt = user_train[user][-1]   ## last item in item seq
            neg_list = np.zeros([aug_size, maxlen+aug_len], dtype=np.int32)
    
            idx = maxlen - 1
            ts = set(user_train[user])  ## targets..!
            for i in reversed(user_train[user][:-1]): 
                seq[idx] = i
                pos[idx] = nxt             ## pos seq
                if (nxt != 0): 
                    for it in range(aug_size):
                        neg_list[it][idx] = random_neq(1, itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break

            orig_train[user] = (seq, pos, neg_list)
        
        return orig_train
        
    def aug_drop(orig_train):
        aug_train = dict()
        aug_len = round(maxlen * do_rate)
        
        for uid in tqdm(range(1, usernum+1), desc='Preparing augmented sample dict'):
            seq = orig_train[uid][0]
            pos = orig_train[uid][1]
            neg_list = orig_train[uid][2]
                    
            for it in reversed(range(aug_size)):
                n_uid = uid * aug_size - it
                n_seq = seq.copy()
                n_pos = pos.copy()
                n_neg = neg_list[it].copy()
                
                # original sequence 살리기
                if it == 0:
                    drop_idx = list(range(aug_len))
                    n_drop_idx = drop_idx
                else:
                    drop_idx = random.sample(range(1, maxlen), aug_len)
                    n_drop_idx = list(map(lambda x: x-1, drop_idx))
                
                n_seq = np.delete(n_seq, drop_idx)
                n_neg = np.delete(n_neg, drop_idx)
                n_pos = np.delete(n_pos, n_drop_idx)
                
                aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
        
        print('aug len & aug size & seq drop rate : ', aug_len, '/ ', aug_size, '/ ', do_rate) 
        print('# of augmented trainining samples: ', len(aug_train)) 
        return aug_train
    
    
    def sample_pad():
        orig_train = dict()
        
        for user in tqdm(range(1, usernum+1), desc='Preparing original sample dict'):
            seq = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            nxt = user_train[user][-1]   ## last item in item seq
            neg_list = np.zeros([aug_size, maxlen], dtype=np.int32)
    
            idx = maxlen - 1
            ts = set(user_train[user])  ## targets..!
            for i in reversed(user_train[user][:-1]):     
                seq[idx] = i
                pos[idx] = nxt             ## pos seq
                if (nxt != 0): 
                    for it in range(aug_size):
                        neg_list[it][idx] = random_neq(1, itemnum + 1, ts) 
                nxt = i
                idx -= 1
                if idx == -1: break
                    
            orig_train[user] = (seq, pos, neg_list)
        
        return orig_train
    
    def aug_pad(orig_train):
        aug_train = dict()
        aug_len = round(maxlen * do_rate)
        
        for uid in tqdm(range(1, usernum+1), desc='Preparing augmented sample dict'):
            seq = orig_train[uid][0]
            pos = orig_train[uid][1]
            neg_list = orig_train[uid][2]
        
            for it in reversed(range(aug_size)):
                n_uid = uid * aug_size - it
                n_seq = seq.copy()
                n_pos = pos.copy()
                
                # original sequence 
                if it == 0:
                    pass
                
                else:
                    drop_idx = random.sample(range(1, maxlen-1), aug_len)
                    #n_drop_idx = list(map(lambda x: x-1, drop_idx))
                
                    for d_idx in drop_idx:
                        n_seq[d_idx] = 0
                
                n_neg = neg_list[it]
                
                aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
        
        print('aug len & aug size & seq drop rate : ', aug_len, '/ ', aug_size, '/ ', do_rate) 
        print('# of augmented trainining samples: ', len(aug_train)) 
        return aug_train
     
    def aug_noise(orig_train):
        aug_train = dict()
        aug_len = 1
        #aug_len = round(maxlen * do_rate)
        
        for uid in tqdm(range(1, usernum+1), desc='Preparing augmented sample dict'):
            seq = orig_train[uid][0]
            pos = orig_train[uid][1]
            neg_list = orig_train[uid][2]
            ts = set(user_train[uid]) 
        
            for it in reversed(range(aug_size)):
                n_uid = uid * aug_size - it
                n_seq = seq.copy()
                n_pos = pos.copy()
                
                # original sequence
                if it == 0:
                    pass
                
                else:
                    noise_idx = np.random.randint(0, maxlen-2)
                    noise_item = random_neq(1, itemnum + 1, ts)
                    #n_drop_idx = list(map(lambda x: x-1, drop_idx))
                    n_seq[:noise_idx+1] = np.concatenate( ( n_seq[1:noise_idx+1], np.array([noise_item]) ), axis=None)
                    n_pos[:noise_idx+1] = np.concatenate( ( n_pos[1:noise_idx+1], np.array(n_pos[noise_idx]) ), axis=None)
                
                n_neg = neg_list[it]
                
                aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
        
        print('aug len & aug size & seq drop rate : ', aug_len, '/ ', aug_size, '/ ', do_rate) 
        print('# of augmented trainining samples: ', len(aug_train)) 
        return aug_train
        
    def aug_redund(orig_train):
        aug_train = dict()
        aug_len = 1
        #aug_len = round(maxlen * do_rate)
        
        for uid in tqdm(range(1, usernum+1), desc='Preparing augmented sample dict'):
            seq = orig_train[uid][0]
            pos = orig_train[uid][1]
            neg_list = orig_train[uid][2]
            ts = set(user_train[uid]) 
        
            for it in reversed(range(aug_size)):
                n_uid = uid * aug_size - it
                n_seq = seq.copy()
                n_pos = pos.copy()
                
                # original sequence
                if it == 0:
                    pass
                
                else:
                    red_idx = np.random.randint(0, maxlen-2)
                    red_item_idx = np.random.randint(0, maxlen-2)
                    red_item = n_seq[red_item_idx]
                    
                    n_seq[:red_idx+1] = np.concatenate( ( n_seq[1:red_idx+1], np.array([red_item]) ), axis=None)
                    n_pos[:red_idx+1] = np.concatenate( ( n_pos[1:red_idx+1], np.array(n_pos[red_item_idx]) ), axis=None)
                
                n_neg = neg_list[it]
                
                aug_train[n_uid] = (uid, n_seq, n_pos, n_neg)
        
        print('aug len & aug size & seq drop rate : ', aug_len, '/ ', aug_size, '/ ', do_rate) 
        print('# of augmented trainining samples: ', len(aug_train)) 
        return aug_train
        
    
    """
    APPLY THE AUGMENTATION STRATEGY DESIGNATED
    """
    if aug_type == 'subset':
        orig_train = sample_subset()
        aug_train = aug_subset(orig_train) 
    elif aug_type == 'slide':
        orig_train = sample_slide()
        aug_train = aug_slide(orig_train) 
    elif aug_type == 'noise':
        orig_train = sample_pad()
        aug_train = aug_noise(orig_train)
    elif aug_type == 'redund':
        orig_train = sample_pad()
        aug_train = aug_redund(orig_train)    
    elif aug_type == 'pad':
        orig_train = sample_pad()
        aug_train = aug_pad(orig_train)
    elif aug_type == 'subst':
        orig_train, training_seqs = sample_emb()
        pre_embed_model = get_pre_emb(training_seqs, hid_dim)
        aug_train = aug_subst(orig_train, pre_embed_model)

    
    aug_usernum = len(aug_train)
    
    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, aug_usernum + 1)
            one_batch.append(aug_train[user])

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10,  aug_type='pad', 
                 aug_size=1, do_rate=0.02, hid_dim=50, rand_seed=1234, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      aug_type,
                                                      aug_size,
                                                      do_rate,
                                                      hid_dim,
                                                      self.result_queue,
                                                      rand_seed
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

            

# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    tot_data = 0
    
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        if fname in ['ml-1m_all']:
            u, i = line.rstrip().split(' ')
        else:
            u, i, ts = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        
        User[u].append(i)
        tot_data += 1

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    
    print('# of user & item:', len(user_train), len(user_test), usernum, itemnum, tot_data)
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_5 = 0.0
    HT_5 = 0.0
    NDCG_1 = 0.0
    HT_1 = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(valid[u][0])
        rated.add(test[u][0])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 1:
            NDCG_1 += 1 / np.log2(rank + 2)
            HT_1 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_1 / valid_user, HT_1 / valid_user, NDCG_5 / valid_user, HT_5 / valid_user, NDCG_10 / valid_user, HT_10 / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_5 = 0.0
    HT_5 = 0.0
    NDCG_1 = 0.0
    HT_1 = 0.0
    
    valid_user = 0.0
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(valid[u][0])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 1:
            NDCG_1 += 1 / np.log2(rank + 2)
            HT_1 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_1 / valid_user, HT_1 / valid_user, NDCG_5 / valid_user, HT_5 / valid_user, NDCG_10 / valid_user, HT_10 / valid_user
