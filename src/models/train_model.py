import scipy.sparse as sparse
import implicit


def fit_implicit_model(df, test_offset = None,
                       factors = 100,
                       regularization = 0.1,
                       iterations = 20,
                       alpha_val = 30,
                       **kwargs):


    # Create a numeric user_id and artist_id column
    df['seq_cat'] = df['seq'].astype("category")
    df['event_info'] = df['event_info'].astype("category")
    df['seq_id'] = df['seq_cat'].cat.codes
    df['event_info_id'] = df['event_info'].cat.codes

    # print('fit_implicit_model:', factors, regularization, iterations, alpha_val, kwargs)

    # create mappings for merge
    seq_map = df[['seq_cat', 'seq_id']].drop_duplicates()
    seq_map.rename(columns = {'seq_cat': 'seq'}, inplace = True) # dstream funcs need 'seq'
    event_info_map = df[['event_info', 'event_info_id']].drop_duplicates()

    # The implicit library expects data as a item-user matrix so we
    # create two matricies, one for fitting the model (item-user)
    # and one for recommendations (user-item)

    # relevance = 'normalized_views' if normalized else 'views'
    if test_offset:
        df_train = df[df.seq < test_offset]
    else:
        df_train = df

    relevance = 'views'
    sparse_item_user = sparse.csr_matrix((df_train[relevance].astype(float),
                                         (df_train['event_info_id'],  df_train['seq_id'])))

    sparse_user_item = sparse.csr_matrix((df[relevance].astype(float),
                                         (df['seq_id'], df['event_info_id'])))

    # Initialize the als model and fit it using the sparse item-user matrix
    model = implicit.als.AlternatingLeastSquares(use_gpu = False,
                                                 factors = factors,
                                                 regularization = regularization,
                                                 iterations = iterations)

    # Calculate the confidence by multiplying it by our alpha value.
    sparse_item_user_conf = (sparse_item_user * alpha_val)#.astype('double')

    #Fit the model
    model.fit(sparse_item_user_conf)

    return model, seq_map, event_info_map, sparse_user_item
