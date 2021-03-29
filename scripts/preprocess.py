# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import pandas as pd
import math

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

def preprocess_criteo(datafile):

    train_path="train.txt"
    # train_path="train.txt"
    train_path = os.path.join(datafile, train_path)
    f1 = open(train_path,'r')
    dic= {}
    # generate three fold.
    # train_x: value
    # train_i: index
    # train_y: label
    f_train_value = open(os.path.join(datafile, 'train_x.txt'),'w')
    f_train_index = open(os.path.join(datafile, 'train_i.txt'),'w')
    f_train_label = open(os.path.join(datafile, 'train_y.txt'),'w')

    num_dense, num_sparse = 13, 26
    num_feature = num_dense + num_sparse
    for i in range(num_feature):
        dic[i] = {}

    cnt_train = 0

    #for debug
    #limits = 10000
    index = [1] * num_sparse
    for line in f1:
        cnt_train +=1
        if cnt_train % 100000 ==0:
            print('now train cnt : %d\n' % cnt_train)
        #if cnt_train > limits:
        #	break
        split = line.strip('\n').split('\t')
        # 0-label, 1-13 numerical, 14-39 category
        for i in range(num_dense, num_feature):
            #dic_len = len(dic[i])
            if split[i+1] not in dic[i]:
            # [1, 0] 1 is the index for those whose appear times <= 10   0 indicates the appear times
                dic[i][split[i+1]] = [1,0]
            dic[i][split[i+1]][1] += 1
            if dic[i][split[i+1]][0] == 1 and dic[i][split[i+1]][1] > 10:
                index[i-num_dense] += 1
                dic[i][split[i+1]][0] = index[i-num_dense]
    f1.close()
    print('total entries :%d\n' % (cnt_train - 1))

    # calculate number of category features of every dimension
    kinds = [num_dense]
    for i in range(num_dense, num_feature):
        kinds.append(index[i-num_dense])
    print('number of dimensions : %d' % (len(kinds)-1))
    print(kinds)

    for i in range(1,len(kinds)):
        kinds[i] += kinds[i-1]
    print(kinds)

    # make new data

    f1 = open(train_path,'r')
    cnt_train = 0
    print('remake training data...\n')
    for line in f1:
        cnt_train +=1
        if cnt_train % 100000 ==0:
            print('now train cnt : %d\n' % cnt_train)
        #if cnt_train > limits:
        #	break
        entry = ['0'] * num_feature
        index = [None] * num_feature
        split = line.strip('\n').split('\t')
        label = str(split[0])
        for i in range(num_dense):
            if split[i+1] != '':
                entry[i] = (split[i+1])
            index[i] = (i+1)
        for i in range(num_dense, num_feature):
            if split[i+1] != '':
                entry[i] = '1'
            index[i] = (dic[i][split[i+1]][0])
        for j in range(num_sparse):
            index[num_dense+j] += kinds[j]
        index = [str(item) for item in index]
        f_train_value.write(' '.join(entry)+'\n')
        f_train_index.write(' '.join(index)+'\n')
        f_train_label.write(label+'\n')
    f1.close()


    f_train_value.close()
    f_train_index.close()
    f_train_label.close()

def preprocess_avazu(datafile):

    train_path = './train.csv'
    f1 = open(train_path, 'r')
    dic = {}
    f_train_value = open('./train_x.txt', 'w')
    f_train_index = open('./train_i.txt', 'w')
    f_train_label = open('./train_y.txt', 'w')
    debug = False
    tune = False
    Bound = [5] * 24

    label_index = 1
    Column = 24

    numr_feat = []
    numerical = [0] * Column
    numerical[label_index] = -1

    cate_feat = []
    for i in range(Column):
        if (numerical[i] == 0):
            cate_feat.extend([i])

    index_cnt = 0
    index_others = [0] * Column
    Max = [0] * Column


    for i in numr_feat:
        index_others[i] = index_cnt
        index_cnt += 1
        numerical[i] = 1
    for i in cate_feat:
        index_others[i] = index_cnt
        index_cnt += 1

    for i in range(Column):
        dic[i] = dict()

    cnt_line = 0
    for line in f1:
        cnt_line += 1
        if (cnt_line == 1): continue # header
        if (cnt_line % 1000000 == 0):
            print ("cnt_line = %d, index_cnt = %d" % (cnt_line, index_cnt))
        if (debug == True):
            if (cnt_line >= 10000):
                break
        split = line.strip('\n').split(',')
        for i in cate_feat:
            if (split[i] != ''):
                if split[i] not in dic[i]:
                    dic[i][split[i]] = [index_others[i], 0]
                dic[i][split[i]][1] += 1
                if (dic[i][split[i]][0] == index_others[i] and dic[i][split[i]][1] == Bound[i]):
                    dic[i][split[i]][0] = index_cnt
                    index_cnt += 1

        if (tune == False):
            label = split[label_index]
            if (label != '0'): label = '1'
            index = [0] * (Column - 1)
            value = ['0'] * (Column - 1)
            for i in range(Column):
                cur = i
                if (i == label_index): continue
                if (i > label_index): cur = i - 1
                if (numerical[i] == 1):
                    index[cur] = index_others[i]
                    if (split[i] != ''):
                        value[cur] = split[i]
                        # Max[i] = max(int(split[i]), Max[i])
                else:
                    if (split[i] != ''):
                        index[cur] = dic[i][split[i]][0]
                        value[cur] = '1'

                if (split[i] == ''):
                    value[cur] = '0'

            f_train_index.write(' '.join(str(i) for i in index) + '\n')
            f_train_value.write(' '.join(value) + '\n')
            f_train_label.write(label + '\n')

    f1.close()
    f_train_index.close()
    f_train_value.close()
    f_train_label.close()
    print ("Finished!")
    print ("index_cnt = %d" % index_cnt)
    # print ("max number for numerical features:")
    # for i in numr_feat:
    #     print ("no.:%d max: %d" % (i, Max[i]))

def preprocess_kdd(datafile):
    #coding=utf-8
    #Email of the author: zjduan@pku.edu.cn
    '''
    0. Click：
    1. Impression（numerical）
    2. DisplayURL: (categorical)
    3. AdID:(categorical)
    4. AdvertiserID:(categorical)
    5. Depth:(numerical)
    6. Position:(numerical)
    7. QueryID:  (categorical) the key of the data file 'queryid_tokensid.txt'.
    8. KeywordID: (categorical)the key of  'purchasedkeyword_tokensid.txt'.
    9. TitleID:  (categorical)the key of 'titleid_tokensid.txt'.
    10. DescriptionID:  (categorical)the key of 'descriptionid_tokensid.txt'.
    11. UserID: (categorical)the key of 'userid_profile.txt'
    12. User's Gender: (categorical)
    13. User's Age: (categorical)
    '''
    train_path = './training.txt'
    f1 = open(train_path, 'r')
    f2 = open('./userid_profile.txt', 'r')
    dic = {}
    f_train_value = open('./train_x.txt', 'w')
    f_train_index = open('./train_i.txt', 'w')
    f_train_label = open('./train_y.txt', 'w')
    debug = False
    tune = False
    Column = 12
    Field = 13

    numr_feat = [1,5,6]
    numerical = [0] * Column
    cate_feat = [2,3,4,7,8,9,10,11]
    index_cnt = 0
    index_others = [0] * (Field + 1)
    Max = [0] * 12
    numerical[0] = -1
    for i in numr_feat:
        index_others[i] = index_cnt
        index_cnt += 1
        numerical[i] = 1
    for i in cate_feat:
        index_others[i] = index_cnt
        index_cnt += 1

    for i in range(Field + 1):
        dic[i] = dict()

    ###init user_dic
    user_dic = dict()

    cnt_line = 0
    for line in f2:
        cnt_line += 1
        if (cnt_line % 1000000 == 0):
            print ("cnt_line = %d, index_cnt = %d" % (cnt_line, index_cnt))
        # if (debug == True):
        #     if (cnt_line >= 10000):
        #         break
        split = line.strip('\n').split('\t')
        user_dic[split[0]] = [split[1], split[2]]
        if (split[1] not in dic[12]):
            dic[12][split[1]] = [index_cnt, 0]
            index_cnt += 1
        if (split[2] not in dic[13]):
            dic[13][split[2]] = [index_cnt, 0]
            index_cnt += 1

    cnt_line = 0
    for line in f1:
        cnt_line += 1
        if (cnt_line % 1000000 == 0):
            print ("cnt_line = %d, index_cnt = %d" % (cnt_line, index_cnt))
        if (debug == True):
            if (cnt_line >= 10000):
                break
        split = line.strip('\n').split('\t')
        for i in cate_feat:
            if (split[i] != ''):
                if split[i] not in dic[i]:
                    dic[i][split[i]] = [index_others[i], 0]
                dic[i][split[i]][1] += 1
                if (dic[i][split[i]][0] == index_others[i] and dic[i][split[i]][1] == 10):
                    dic[i][split[i]][0] = index_cnt
                    index_cnt += 1

        if (tune == False):
            label = split[0]
            if (label != '0'): label = '1'
            index = [0] * Field
            value = ['0'] * Field
            for i in range(1, 12):
                if (numerical[i] == 1):
                    index[i - 1] = index_others[i]
                    if (split[i] != ''):
                        value[i - 1] = split[i]
                        Max[i] = max(int(split[i]), Max[i])
                else:
                    if (split[i] != ''):
                        index[i - 1] = dic[i][split[i]][0]
                        value[i - 1] = '1'

                if (split[i] == ''):
                    value[i - 1] = '0'
                if (i == 11 and split[i] == '0'):
                    value[i - 1] = '0'
            ### gender and age
            if (split[11] == '' or (split[11] not in user_dic)):
                index[12 - 1] = index_others[12]
                value[12 - 1] = '0'
                index[13 - 1] = index_others[13]
                value[13 - 1] = '0'
            else:
                index[12 - 1] = dic[12][user_dic[split[11]][0]][0]
                value[12 - 1] = '1'
                index[13 - 1] = dic[13][user_dic[split[11]][1]][0]
                value[13 - 1] = '1'

            f_train_index.write(' '.join(str(i) for i in index) + '\n')
            f_train_value.write(' '.join(value) + '\n')
            f_train_label.write(label + '\n')

    f1.close()
    f_train_index.close()
    f_train_value.close()
    f_train_label.close()
    print ("Finished!")
    print ("index_cnt = %d" % index_cnt)
    print ("max number for numerical features:")
    for i in numr_feat:
        print ("no.:%d max: %d" % (i, Max[i]))

def _load_data(_nrows=None, debug = False, datafile=""):

    TRAIN_X = os.path.join(datafile, 'train_x.txt')
    TRAIN_Y = os.path.join(datafile, 'train_y.txt')

    print(TRAIN_X)
    print(TRAIN_Y)
    train_x = pd.read_csv(TRAIN_X,header=None,sep=' ',nrows=_nrows, dtype=np.float)
    train_y = pd.read_csv(TRAIN_Y,header=None,sep=' ',nrows=_nrows, dtype=np.int32)


    train_x = train_x.values
    train_y = train_y.values.reshape([-1])


    print('data loading done!')
    print('training data : %d' % train_y.shape[0])


    assert train_x.shape[0]==train_y.shape[0]

    return train_x, train_y


def save_x_y(fold_index, train_x, train_y, datafile):
    train_x_name = "train_x.npy"
    train_y_name = "train_y.npy"
    _get = lambda x, l: [x[i] for i in l]
    for i in range(len(fold_index)):
        print("now part %d" % (i+1))
        part_index = fold_index[i]
        Xv_train_, y_train_ = _get(train_x, part_index), _get(train_y, part_index)
        save_dir_Xv = os.path.join(datafile, "part" + str(i+1))
        save_dir_y = os.path.join(datafile, "part" + str(i+1))
        if (os.path.exists(save_dir_Xv) == False):
            os.makedirs(save_dir_Xv)
        if (os.path.exists(save_dir_y) == False):
            os.makedirs(save_dir_y)
        save_path_Xv  = os.path.join(save_dir_Xv, train_x_name)
        save_path_y  = os.path.join(save_dir_y, train_y_name)
        np.save(save_path_Xv, Xv_train_)
        np.save(save_path_y, y_train_)


def save_i(fold_index, datafile):
    _get = lambda x, l: [x[i] for i in l]

    TRAIN_I = os.path.join(datafile, 'train_i.txt')
    train_i = pd.read_csv(TRAIN_I,header=None,sep=' ',nrows=None, dtype=np.int32)
    train_i = train_i.values
    feature_size = train_i.max() + 1
    print ("feature_size = %d" % feature_size)
    feature_size = [feature_size]
    feature_size = np.array(feature_size)
    np.save(os.path.join(datafile, "feature_size.npy"), feature_size)

    # pivot = 40000000

    # test_i = train_i[pivot:]
    # train_i = train_i[:pivot]

    # print("test_i size: %d" % len(test_i))
    print("train_i size: %d" % len(train_i))

    # np.save("../data/test/test_i.npy", test_i)

    for i in range(len(fold_index)):
        print("now part %d" % (i+1))
        part_index = fold_index[i]
        Xi_train_ = _get(train_i, part_index)
        save_path_Xi  = os.path.join(datafile, "part" + str(i+1), 'train_i.npy')
        np.save(save_path_Xi, Xi_train_)



def stratifiedKfold(datafile):


    train_x, train_y = _load_data(datafile=datafile)
    print('loading data done!')

    folds = list(StratifiedKFold(n_splits=10, shuffle=True,
                             random_state=2018).split(train_x, train_y))

    fold_index = []
    for i,(train_id, valid_id) in enumerate(folds):
        fold_index.append(valid_id)

    print("fold num: %d" % (len(fold_index)))

    fold_index = np.array(fold_index)
    np.save(os.path.join(datafile, "fold_index.npy"), fold_index)

    save_x_y(fold_index, train_x, train_y, datafile=datafile)
    print("save train_x_y done!")

    fold_index = np.load(os.path.join(datafile, "fold_index.npy"), allow_pickle=True)
    save_i(fold_index, datafile=datafile)
    print("save index done!")



def scale(x):
    if x > 2:
        x = int(math.log(float(x))**2)
    return x



def scale_dense_feat(datafile, dataset_name):


    if args.dataset_name == "criteo":
        num_dense = 13
    elif args.dataset_name == "avazu":
        return True
    elif args.dataset_name == "kdd":
        num_dense = 3

    for i in range(1,11):
        print('now part %d' % i)
        data = np.load(os.path.join(datafile, 'part'+str(i), 'train_x.npy'),  allow_pickle=True)
        part = data[:,:num_dense]
        for j in range(part.shape[0]):
            if j % 100000 ==0:
                print(j)
            part[j] = list(map(scale, part[j]))
        np.save(os.path.join(datafile, 'part' + str(i), 'train_x2.npy'), data)



def print_shape(name, var):
    print("Shape of {}: {}".format(name, var.shape))


def check_existing_file(filename, force):
    if os.path.isfile(filename):
        print("file {} already exists!".format(filename))
        if not force:
            raise ValueError("aborting, use --force if you want to processed")
        else:
            print("Will override the file!")

def sample_data(args):
    output_data_file = "{}{}.npz".format(args.data_file, args.save_filename)
    check_existing_file(output_data_file, args.force)

    data = np.load(args.sample_data_file,  allow_pickle=True)
    X_cat, X_int, y = data["X_cat"], data["X_int"], data["y"]
    print_shape("X_cat", X_cat)
    print_shape("X_int", X_int)
    print_shape("y", y)
    print("total number of data points: {}".format(len(y)))

    print(
        "saving first {} data points to {}{}.npz".format(
            args.num_samples, args.data_file, args.save_filename
        )
    )
    np.savez_compressed(
        "{}{}.npz".format(args.data_file, args.save_filename),
        X_int=X_int[0 : args.num_samples, :],
        X_cat=X_cat[0 : args.num_samples, :],
        y=y[0 : args.num_samples],
    )



def compress_ids(feature, raw_to_new={}):
    if raw_to_new is None:
        start_idx = 1
        raw_to_new = {}
    else:
        start_idx = 0

    for i in range(len(feature)):
        if feature[i] not in raw_to_new:
            raw_to_new[feature[i]] = len(raw_to_new) + start_idx
        feature[i] = raw_to_new[feature[i]]
    return raw_to_new

def final_preprocess(datafile):
    X_int = []
    X_cat = []
    y = []
    missing_sparse = []


    if args.dataset_name == "criteo":
        num_dense, num_sparse = 13, 26
        TRAIN_X = "train_x2.npy"
    elif args.dataset_name == "avazu":
        num_dense, num_sparse = 0, 23
        TRAIN_X = "train_x.npy"
    elif args.dataset_name == "kdd":
        num_dense, num_sparse = 3, 10
        TRAIN_X = "train_x2.npy"

    TRAIN_Y = "train_y.npy"
    TRAIN_I = "train_i.npy"

    for i in [3,4,5,6,7,8,9,10,2,1]:#range(1,11): # todo
        f = np.load(os.path.join(datafile, "part" + str(i), TRAIN_I), "r",  allow_pickle=True)
        g = np.load(os.path.join(datafile, "part" + str(i), TRAIN_X), "r",  allow_pickle=True)
        h = np.load(os.path.join(datafile, "part" + str(i), TRAIN_Y), "r",  allow_pickle=True)

        X_int_split = np.array(g[:, 0:num_dense])
        X_cat_split = np.array(f[:, num_dense:])
        y_split = h
        missing_sparse_split = np.array(g[:,0:])

        indices = np.arange(len(y_split))
        indices = np.random.permutation(indices)

        # shuffle data
        X_cat_split = X_cat_split[indices]
        X_int_split = X_int_split[indices]
        y_split = y_split[indices].astype(np.float32)
        missing_sparse_split = missing_sparse_split[indices]

        X_int.append(X_int_split)
        X_cat.append(X_cat_split)
        y.append(y_split)
        missing_sparse.append(missing_sparse_split)

    X_int = np.concatenate(X_int)
    X_cat = np.concatenate(X_cat)
    y = np.concatenate(y)
    missing_sparse = np.concatenate(missing_sparse)

    print("expected feature size", X_cat.max() + 1)

    flat = X_cat.flatten()

    fset = set(flat)
    print("expected size", len(fset))


    missing_sparse_maps = []

    for i in range(num_sparse):
        missing_slice = missing_sparse[:,i]
        if 0 in missing_slice:
            locs = np.where(missing_slice==0)[0]
            missing_sparse_maps.append({X_cat[locs[0],i]:0})
        else:
            missing_sparse_maps.append(None)

    raw_to_new_ids = []
    for i in range(X_cat.shape[1]):
        print("compressing the ids for the {}-th feature.".format(i))
        raw_to_new_ids.append(compress_ids(X_cat[:, i], missing_sparse_maps[i]))


    total = 0
    hashsizes = []
    for i in range(len(raw_to_new_ids)):
        hashsize = max(raw_to_new_ids[i].values())+1 # 1 is for the zero
        hashsizes.append(hashsize)
        print("sparse_" + str(i),"\t", hashsize)
        total += hashsize


    if args.dataset_name == "criteo":
        hashsize_filename = "criteo_hashsizes.npy"
        finaldata_filename = "criteo_processed.npz"
    elif args.dataset_name == "avazu":
        hashsize_filename = "avazu_hashsizes.npy"
        finaldata_filename = "avazu_processed.npz"
    elif args.dataset_name == "kdd":
        hashsize_filename = "kdd2012_hashsizes.npy"
        finaldata_filename = "kdd2012_processed.npz"
    np.save(os.path.join(datafile, hashsize_filename), np.array(hashsizes))
    np.savez_compressed(os.path.join(datafile, finaldata_filename), X_int=X_int, X_cat=X_cat, y=y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse Data")
    parser.add_argument("--dataset-name",  default="criteo", choices=["criteo", "avazu", "kdd"])
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--sample-data-file", type=str, default="")
    parser.add_argument("--save-filename", type=str, default="")
    parser.add_argument("--mode", type=str, default="raw")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    if args.mode == "raw":
        print(
            "Load raw data and parse (compress id to consecutive space, "
            "shuffle within ds) and save it."
        )

        if args.dataset_name == "criteo":
            preprocess_criteo(datafile=args.data_file)
        elif args.dataset_name == "avazu":
            preprocess_avazu(datafile=args.data_file)
        elif args.dataset_name == "kdd":
            preprocess_kdd(datafile=args.data_file)

        print("Start stratifiedKfold!")
        stratifiedKfold(datafile=args.data_file)

        print("Start scaling!")
        scale_dense_feat(datafile=args.data_file, dataset_name=args.dataset_name)

        print("Final preprocessing stage!")
        final_preprocess(datafile=args.data_file)

        print("Finish data preprocessing!")

    elif args.mode == "sample":
        print("Load processed data and take the first K data points and save it.")
        sample_data(args)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))
