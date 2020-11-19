from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import pandas as pd
import numpy as np
import os


def read_processed(train_fn, test_fn):

    project_dir = Path(__file__).resolve().parents[2]
    processed_data_dir = os.path.join(project_dir, os.environ.get("PROCESSED_DATA_DIR"))

    # train_fn = 'train_dataset.pkl'
    train_fp = os.path.join(processed_data_dir, train_fn)
    train = pd.read_pickle(train_fp)

    # test_fn = 'test_dataset.pkl'
    test_fp = os.path.join(processed_data_dir, test_fn)
    test = pd.read_pickle(test_fp)

    return train, test

def shrink_and_split(train, keep_train = None, validation = None):

    print('train shape:', train.shape)

    if keep_train:
        unique_seqs = train.seq.unique()
        selected_seqs = np.random.choice(unique_seqs,
                                         size = int(len(unique_seqs) * keep_train),
                                         replace = False)
        train = train[train.seq.isin(selected_seqs)].copy()

    if validation:
        unique_seqs = train.seq.unique()
        selected_seqs = np.random.choice(unique_seqs,
                                         size = int(len(unique_seqs) * validation),
                                         replace = False)
        test = train[train.seq.isin(selected_seqs)].copy()
        test.drop('item_bought', axis = 1, inplace = True)
        test = test[test.event_type != 'buy'] # change to 'buy' if this comes from dataprep
        train = train[~train.seq.isin(selected_seqs)].copy()

    print('train/test shapes:', train.shape, test.shape)

    return train, test


def join_prepare_train_test(df_train, df_test,
                            buy_weight = None, return_search = False,
                            drop_timezone = True,
                            **kwargs):
    # print('join_prepare_train_test:,', buy_weight, kwargs)

    test_offset = df_train.seq.max() + 1
    df_test_copy = df_test.copy()
    df_test_copy.seq = df_test_copy.seq + test_offset
    test_shifted_seq_vals = np.sort(np.unique(df_test_copy.seq))
    df = pd.concat([df_train, df_test_copy])
    df['event_type'] = df.event_type.fillna('buy') # not needed if 'buy' is filled in dataprep

    buy_idx = df.event_type == 'buy'
    search_idx = df.event_type == 'search'

    if buy_weight:
        df_buy = df[buy_idx].copy()
        df_buy.drop('event_info', axis = 1, inplace = True)
        df_buy.rename(columns = {'item_bought': 'event_info'}, inplace = True)
        df_buy = df_buy.drop(['event_timestamp', 'event_type', 'time_diff'], axis = 1)
        df_buy['views'] = buy_weight
        df_buy['event_type'] = 'buy'

    if return_search:
        df_search = df[search_idx].copy()
        df_search.drop(['event_timestamp', 'time_diff', 'item_bought'], axis = 1, inplace = True)

    df = df[~(buy_idx | search_idx)].copy() # only views
    df = df.drop(['event_timestamp', 'item_bought', 'time_diff'], axis = 1)
    df = df.groupby(['seq', 'event_info']).event_type.count().reset_index()
    df.rename(columns = {'event_type': 'views'}, inplace = True) # fixing col name
    df['event_type'] = 'view'

    if buy_weight:
        df = pd.concat([df, df_buy]).sort_values('seq')
        #df['event_info'] = df['event_info'].astype(int)

    if return_search:
        df = pd.concat([df, df_search])

    if drop_timezone and ('timezone' in df):
        df = df.drop(columns = ['timezone'])

    df = df.sort_values(['seq', 'event_type'], ascending = [True, False])

    # df['normalized_views'] = df.groupby('seq').views.transform(lambda x: x/sum(x))

    return test_offset, test_shifted_seq_vals, df
