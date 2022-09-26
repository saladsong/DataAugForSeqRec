import gzip
import csv
from collections import defaultdict
from datetime import datetime
import time

countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

dataset_name = 'ml-1m'
dataset = 'ml-1m_total.txt'  ## with header & no ratings
#dataset_name = 'Video'
#dataset = 'Video_total.txt'
#dataset_name = 'gowalla'
#dataset = 'gowalla_total.txt'
new_f = open(dataset_name+'.txt', 'w')

with open(dataset, "r") as f:
#for l in parse('reviews_' + dataset_name + '.json.gz'):
    reader = csv.DictReader(f, delimiter=',')
    for data in reader:
        line += 1
        uid = data['user_id']
        item_id = data['item_id']
        ts = data['timestamp']
        new_f.write(" ".join([uid, item_id, str(ts)]) + ' \n')
        countU[uid] += 1
        countP[item_id] += 1
new_f.close()

print('1st phase done')

item_map = dict()
usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
with open(dataset, "r") as f:
    reader = csv.DictReader(f, delimiter=',')
    for data in reader:
        uid = data['user_id']
        item_id = data['item_id']  # from original dataset
        ts = data['timestamp']
        tstamp = int(ts)  # ml1m, amazon
        #tstamp= time.mktime(datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ').timetuple()) # gowalla

        #if (countU[uid] < 5) or (countP[item_id] < 5):
        if (countU[uid] < 5) or (countP[item_id] < 5) or (int(uid) % 10 > 0):  # for 10% fraction sub-dataset
            continue

        if uid in usermap:
            userid = usermap[uid]
        else:
            usernum += 1
            userid = usernum
            usermap[uid] = userid
            User[userid] = []
        if item_id in itemmap:
            itemid = itemmap[item_id]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[item_id] = itemid
        
        User[userid].append([tstamp, itemid])

# sort reviews in User according to time
for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print('user, item, interaction #:' , usernum, itemnum, line)

f = open(dataset_name+'_dime.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d %d\n' % (user, i[1], i[0]))
f.close()
