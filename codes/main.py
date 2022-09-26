import os
import time
import torch
import argparse

from model import SASRec
from tqdm import tqdm
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_type', required=True)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--aug_type', default='subset', type=str)
parser.add_argument('--aug_size', default=10, type=int)
parser.add_argument('--do_rate', default=0.1, type=float)
parser.add_argument('--rand_seed', default=123, type=int)
parser.add_argument('--hidden_units', default=100, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=401, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--train_dir', default='default')
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
print(args)
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

##
fname = args.dataset + '_' + args.train_type
dataset = data_partition(fname)

[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = (len(user_train)*args.aug_size) // args.batch_size 
print('num batch as # of iter: ', num_batch)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, args.aug_type + '_' + str(args.train_type)  + '_' + str(args.rand_seed) + '_log.txt'), 'w')
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, aug_type=args.aug_type, aug_size=args.aug_size, do_rate=args.do_rate, hid_dim=args.hidden_units, rand_seed=args.rand_seed,  n_workers=1)
model = SASRec(usernum, itemnum, itemnum, args).to(args.device) 
model.train() # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        print('failed loading state_dicts, pls check file path: ', end="")
        print(args.state_dict_path)
        print('pdb enabled for your quick check, pls type exit() if you do not need it')
        import pdb; pdb.set_trace()
        

if args.inference_only:
    model.eval()
    t_test = evaluate(model, dataset, args)
    print('epoch:%d, time: %f(s), test (NDCG@1: %.4f, HR@1: %.4f)' % (epoch, T, t_test[0], t_test[1]))
    print('epoch:%d, time: %f(s), test (NDCG@5: %.4f, HR@5: %.4f)' % (epoch, T, t_test[2], t_test[3]))
    print('epoch:%d, time: %f(s), test (NDCG@10: %.4f, HR@10: %.4f)' % (epoch, T, t_test[4], t_test[5]))

# ce_criterion = torch.nn.CrossEntropyLoss()
# https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

T = 0.0
t0 = time.time()

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only: break # just to decrease identition
    for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
        # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
        adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        loss.backward()
        adam_optimizer.step()
        
    if (epoch % 5) == 0:
        print("loss in epoch {}: {}".format(epoch, loss.item()))

    if epoch % 10 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_test = evaluate(model, dataset, args)
        t_valid = evaluate_valid(model, dataset, args)
        #print('epoch:%d, time: %f(s), valid (NDCG@1: %.4f, HR@1: %.4f), test (NDCG@1: %.4f, HR@1: %.4f)'
        #        % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
        print('epoch:%d, time: %f(s), valid (NDCG@5: %.4f, HR@5: %.4f), test (NDCG@5: %.4f, HR@5: %.4f)'
                % (epoch, T, t_valid[2], t_valid[3], t_test[2], t_test[3]))
        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                % (epoch, T, t_valid[4], t_valid[5], t_test[4], t_test[5]))

        #f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        f.write( 'epoch:%d, time: %f(s), valid (NDCG@5: %.4f, HR@5: %.4f), test (NDCG@5: %.4f, HR@5: %.4f) \n'
                % (epoch, T, t_valid[2], t_valid[3], t_test[2], t_test[3]) )
        f.write( 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f) \n'
                % (epoch, T, t_valid[4], t_valid[5], t_test[4], t_test[5]) )
        f.write('..................................................................................\n')
        f.flush()
        t0 = time.time()
        model.train()

    if epoch == args.num_epochs:
        folder = args.dataset + '_' + args.train_dir
        fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        torch.save(model.state_dict(), os.path.join(folder, fname))

f.close()
sampler.close()
print(args)
print("Done")
