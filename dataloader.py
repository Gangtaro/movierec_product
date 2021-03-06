import sys, os, random
import logging
import pandas as pd
import sys
import logging
import torch
from tqdm import tqdm 
from time import time
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, path = '../data/', mode = 'train'):
        self.path = path # default: '../data/'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
        data_path = os.path.join(self.path, 'train/train_ratings.csv')
        genre_path = os.path.join(self.path, 'train/genres.tsv')
        df = pd.read_csv(data_path)
        genre_data = pd.read_csv(genre_path, sep='\t')

        self.ratings_df = df.copy() # for submission
        self.n_train = len(df)

        item_ids = df['item'].unique() # 아이템 고유 번호 리스트
        user_ids = df['user'].unique() # 유저 고유 번호 리스트
        self.n_items, self.n_users = len(item_ids), len(user_ids)
        
        # user, item indexing
        self.item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids) # item re-indexing (0~num_item-1) ; 아이템을 1부터 설정하는이유? 0을 아무것도 아닌 것으로 blank 하기 위해서
        self.user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': self.item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': self.user2idx[user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        genre_data = df.merge(genre_data, on = 'item').copy()
        del df['item'], df['user'], genre_data['item'], genre_data['user']

        self.exist_items = list(df['item_idx'].unique())
        self.exist_users = list(df['user_idx'].unique())

        t1 = time()
        self.train_items, self.valid_items = {}, {}
        
        items = df.groupby("user_idx")["item_idx"].apply(list) # 유저 아이디 상관 없이, 순서대로 
        if mode == 'train':
            print('Creating interaction Train/ Vaild Split...')
            for uid, item in enumerate(items):            
                num_u_valid_items = min(int(len(item)*0.125), 10) # 유저가 소비한 아이템의 12.5%, 그리고 최대 10개의 데이터셋을 무작위로 Validation Set으로 활용한다.

                ####### method-3 : hybrid ####### 마지막꺼:무작위= 6:4
                num_random = np.floor(num_u_valid_items*0.6).astype(int) # 홀수일때는, 무작위로 뽑는것이 1개 더 많게
                num_last = int(num_u_valid_items - num_random)
                last_items = item[-num_last:]
                random_items = np.random.choice(item[:-num_last], size=num_random, replace=False).tolist()
                u_valid_items = random_items + last_items
                self.valid_items[uid] = u_valid_items
                self.train_items[uid] = list(set(item) - set(u_valid_items))

            self.train_data = pd.concat({k: pd.Series(v) for k, v in self.train_items.items()}).reset_index(0)
            self.train_data.columns = ['user', 'item']

            self.valid_data = pd.concat({k: pd.Series(v) for k, v in self.valid_items.items()}).reset_index(0)
            self.valid_data.columns = ['user', 'item']
        
        if mode == 'train_all': #else
            print('Preparing interaction all train set')
            self.train_data = pd.DataFrame()
            self.train_data['user'] = df['user_idx']
            self.train_data['item'] = df['item_idx']

        print('Train/Vaild Split Complete. Takes in', time() - t1, 'sec')
        
        rows, cols = self.train_data['user'], self.train_data['item']
        self.train_input_data = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))
        self.train_input_data = self.train_input_data.toarray()


        print('Making Genre filter ... ')
        genre2item = genre_data.groupby('genre')['item_idx'].apply(set).apply(list)

        genre_data_freq = genre_data.groupby('user_idx')['genre'].value_counts(normalize=True)
        genre_data_freq_over_5p = genre_data_freq[genre_data_freq > 0.003].reset_index('user_idx')
        genre_data_freq_over_5p.columns = ['user_idx', 'tobedroped']
        genre_data_freq_over_5p = genre_data_freq_over_5p.drop('tobedroped', axis = 1).reset_index()
        user2genre = genre_data_freq_over_5p.groupby('user_idx')['genre'].apply(set).apply(list)

        genre2item_dict = genre2item.to_dict()
        all_set_genre = set(genre_data['genre'].unique())
        user_genre_filter_dict = {}
        for user, genres in tqdm(enumerate(user2genre)):
            unseen_genres = all_set_genre - set(genres) # set
            unseen_genres_item = set(sum([genre2item_dict[genre] for genre in unseen_genres], []))
            user_genre_filter_dict[user] = pd.Series(list(unseen_genres_item), dtype=np.int32)

        user_genre_filter_df = pd.concat(user_genre_filter_dict).reset_index(0)
        user_genre_filter_df.columns = ['user', 'item']
        user_genre_filter_df.index = range(len(user_genre_filter_df))

        rows, cols = user_genre_filter_df['user'], user_genre_filter_df['item']
        self.user_genre_filter = sp.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float32',
            shape=(self.n_users, self.n_items))
        self.user_genre_filter = self.user_genre_filter.toarray()

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_input_data[idx,:]
