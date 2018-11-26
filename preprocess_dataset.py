from collections import defaultdict
import pandas as pd
import json

def order_by_timestamp(dataset_path):
    ratings = pd.read_csv('data/%s'%dataset_path, sep='::').sort_values(['userId', 'timestamp'])
    ratings.to_csv('data/%s'%dataset_path, sep=',')
    print('Dataset ordered by userId and timestamp')

def data_partition(fname, store=True):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_val = {}
    user_test = {}
    # assume user/item index starting from 1
    with open('data/%s' % fname, 'r') as file:
        file.readline()
        for line in file.readlines():
            u, i = line.rstrip().split(',')[1:3]
            u = int(u)
            i = int(i)
            # store max userId
            usernum = max(u, usernum)
            # store max itemId
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nratings = len(User[user])
        if nratings < 3:
            user_train[user] = User[user]
            user_val[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_val[user] = []
            user_val[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    if store:
        with open('data/user_train.json', 'w') as outfile: json.dump(user_train, outfile)
        with open('data/user_val.json', 'w') as outfile: json.dump(user_val, outfile)
        with open('data/user_test.json', 'w') as outfile: json.dump(user_test, outfile)
        with open('data/usernum.txt', 'w') as outfile: json.dump(usernum, outfile)
        with open('data/itemnum.txt', 'w') as outfile: json.dump(itemnum, outfile)
    return [user_train, user_val, user_test, usernum, itemnum]

def main():
    order_by_timestamp('ratings.csv')
    data_partition('ratings.csv')

if __name__ == '__main__':
    main()
