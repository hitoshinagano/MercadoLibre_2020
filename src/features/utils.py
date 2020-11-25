from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


project_dir = Path(__file__).resolve().parents[2]
processed_data_dir = os.path.join(project_dir, os.environ.get("PROCESSED_DATA_DIR"))
raw_data_dir = os.path.join(project_dir, os.environ.get("RAW_DATA_DIR"))

def proc_dataset(df):
    number_of_batches = len(df) // 50
    proc_df = list()
    for df_p in tqdm(np.array_split(df, number_of_batches)):
        if 'item_bought' in df_p:
            df_p = pd.concat([df_p.user_history.apply(pd.Series), df_p.item_bought], axis = 1).stack()
            train_dataset = True
        else:
            df_p = df_p.user_history.apply(pd.Series).stack()
            train_dataset = False

        df_p = df_p.apply(pd.Series)
        df_p.reset_index(inplace = True)
        df_p.drop(columns = 'level_1', inplace = True)

        if train_dataset:
            new_columns = {0: 'item_bought', 'level_0': 'seq'}
            df_p['event_type'] = df_p.event_type.fillna('buy')
        else:
            new_columns = {'level_0': 'seq'}

        df_p.rename(columns = new_columns, inplace = True)

        # df_p['timezone'] = df_p.event_timestamp.str[-4:]
        df_p['event_timestamp'] = pd.to_datetime(df_p.event_timestamp.str[:-9])
        df_p['time_diff'] = df_p.groupby('seq').event_timestamp.diff().dt.seconds

        # if train_dataset:
        proc_df.append(df_p)

    proc_df = pd.concat(proc_df)
    return proc_df

def get_lang_from_views(df):

    # read item_domain.pkl
    item_domain_fn = 'item_domain.pkl'
    item_domain_fp = os.path.join(processed_data_dir, item_domain_fn)
    if not os.path.exists(item_domain_fp):
        print('run EDA_dataprep.ipynb first to create item_domain.pkl')
        assert False
    item_domain = pd.read_pickle(item_domain_fp)
    item_domain['lang_domain'] = item_domain.category_id.str[:3].replace({'MLM': 'es', 'MLB': 'pt'})

    # get language from most prevalent viewed item domains
    lang = df[df.event_type == 'view'].copy()
    lang['event_info'] = lang.event_info.astype(int)
    lang = lang[['seq', 'event_info']]
    lang = pd.merge(lang, item_domain[['item_id', 'lang_domain']], how = 'left',
                     left_on = 'event_info', right_on = 'item_id')
    lang = lang.groupby('seq').lang_domain.value_counts()
    lang = lang.unstack().fillna(0).idxmax(axis = 1)
    lang = lang.reset_index().rename(columns = {0: 'lang_seq'})

    # merge back into df by seq_id
    df = pd.merge(df, lang, how = 'left')

    return df

def get_lang_from_nlp(df):
    """use langdetect on search strings to detect lang"""
    pass

def save_true_labels(df, true_fn = 'true.pkl'):
    true_fp = os.path.join(processed_data_dir, true_fn)
    true_df = df[(df.event_type.isnull()) | (df.event_type == 'buy')]
    true_df = true_df[['seq', 'item_bought']]
    true_df.to_pickle(true_fp)

def read_raw_save_processed(raw_fn = 'train_dataset.jl.gz', processed_fn = 'train_dataset.pkl',
              force_process = False, nrows = None, add_lang = True):

    processed_fp = os.path.join(processed_data_dir, processed_fn)
    if os.path.exists(processed_fp) and not force_process:
        processed = pd.read_pickle(processed_fp)
    else:
        raw = pd.read_json(os.path.join(raw_data_dir, raw_fn), lines = True, nrows = nrows)
        raw['len_events'] = raw.user_history.str.len()
        raw.sort_values('len_events', inplace = True)
        raw.drop('len_events', axis = 1, inplace = True)
        processed = proc_dataset(raw)
        if add_lang:
            processed = get_lang_from_views(processed)
            # call get_lang_from_nlp to detect rows with lang_seq = NaNs
        processed.to_pickle(processed_fp)

    if 'item_bought' in processed:
        processed.item_bought = processed.item_bought.fillna(method = 'backfill').astype(int)
        processed['in_nav'] = processed.item_bought == processed.event_info

    return processed

def read_processed(train_fn, test_fn, keep_train = None, validation = None,
                   lang = 'both'):

    train_fp = os.path.join(processed_data_dir, train_fn)
    train = pd.read_pickle(train_fp)

    if validation is None:
        test_fp = os.path.join(processed_data_dir, test_fn)
        test = pd.read_pickle(test_fp)
    else:
        train, test = shrink_and_split(train,
                                       keep_train = keep_train,
                                       validation = validation)

    for df in [train, test]:
        if 'timezone' in df:
            df.drop(columns = 'timezone', inplace = True)

        if lang == 'es':
            df['lang_seq'] = df.lang_seq.fillna('pt') # if lang not detected make it 'pt'
        else:
            df['lang_seq'] = df.lang_seq.fillna('pt')

    if lang != 'both':
        print('lang', lang)
        train = train[train.lang_seq == lang]
        test = test[test.lang_seq == lang]

    return train, test

def shrink_and_split(train, keep_train = None, validation = None):

    print('initial train shape:', train.shape)

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


# one_hour = np.timedelta64(1, 'h')
# def get_event_weights_h(x):
#     last_ts = x.values[-1]
#     h = (last_ts - x) / one_hour
#     return (1 - alpha) **  h

alpha = 0.02 # 0.1 => 5d : 0.6
one_minus_alpha = 1 - alpha
one_day = np.timedelta64(1, 'D')

def get_event_weights_d(x):
    last_ts = x.values[-1]
    d = (last_ts - x) / one_day
    return one_minus_alpha ** d.round(0)

def join_prepare_train_test(df_train, df_test,
                            buy_weight = None, return_search = False,
                            drop_timezone = True, just_concat = False,
                            extra_weight = 200, lang = 'pt', **kwargs):
    # print('join_prepare_train_test:,', buy_weight, kwargs)
    # breakpoint()
    if isinstance(df_train, str) and isinstance(df_test, str):
        df_train, df_test = read_processed(df_train, df_test, lang = lang)

    test_offset = df_train.seq.max() + 1
    df_test_copy = df_test.copy()
    df_test_copy.seq = df_test_copy.seq + test_offset
    test_shifted_seq_vals = np.sort(np.unique(df_test_copy.seq))
    df = pd.concat([df_train, df_test_copy])
    df['event_type'] = df.event_type.fillna('buy') # not needed if 'buy' is filled in dataprep

    if just_concat:
        if drop_timezone and ('timezone' in df):
            df = df.drop(columns = ['timezone'])
        return test_offset, test_shifted_seq_vals, df

    buy_idx = df.event_type == 'buy'
    search_idx = df.event_type == 'search'

    if buy_weight:
        df_buy = df[buy_idx].copy()
        df_buy.drop('event_info', axis = 1, inplace = True)
        df_buy.rename(columns = {'item_bought': 'event_info'}, inplace = True)
        df_buy = df_buy.drop(['event_timestamp', 'event_type', 'time_diff'], axis = 1)
        df_buy['views'] = buy_weight #+ extra_weight * df_buy['in_nav_pred']
        df_buy['event_type'] = 'buy'

    if return_search:
        df_search = df[search_idx].copy()
        df_search.drop(['event_timestamp', 'time_diff', 'item_bought'], axis = 1, inplace = True)

    df = df[~(buy_idx | search_idx)].copy() # only views
    # df['event_weights'] = df.groupby('seq').event_timestamp.transform(get_event_weights_d)
    df = df.drop(['event_timestamp', 'item_bought', 'time_diff'], axis = 1)
    df = df.groupby(['seq', 'event_info']).event_type.count().reset_index() # simple count
    df.rename(columns = {'event_type': 'views'}, inplace = True) # fixing col name - simple count
    # df = df.groupby(['seq', 'event_info']).event_weights.sum().reset_index()
    # df.rename(columns = {'event_weights': 'views'}, inplace = True) # fixing col name
    df['event_type'] = 'view'

    if buy_weight:
        df = pd.concat([df, df_buy]).sort_values('seq')
        #df['event_info'] = df['event_info'].astype(int)

    if return_search:
        df = pd.concat([df, df_search])

    if drop_timezone and ('timezone' in df):
        df = df.drop(columns = ['timezone'])

    df = df.sort_values(['seq', 'event_type'], ascending = [True, False])

    # df.drop(['lang_seq', 'in_nav', 'in_nav_pred'], axis = 1, inplace = True)
    df.drop(['lang_seq'], axis = 1, inplace = True)

    # df['normalized_views'] = df.groupby('seq').views.transform(lambda x: x/sum(x))

    return test_offset, test_shifted_seq_vals, df
