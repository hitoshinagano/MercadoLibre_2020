import scipy.sparse as sparse
import implicit


def fit_implicit_model(df,
                       factors = 100,
                       regularization = 0.1,
                       iterations = 20,
                       alpha_val = 30,
                       **kwargs):

    # print('fit_implicit_model:', factors, regularization, iterations, alpha_val, kwargs)

    # Create a numeric user_id and artist_id column
    df['seq'] = df['seq'].astype("category")
    df['event_info'] = df['event_info'].astype("category")
    df['seq_id'] = df['seq'].cat.codes
    df['event_info_id'] = df['event_info'].cat.codes

    # create mappings for merge
    seq_map = df[['seq', 'seq_id']].drop_duplicates()
    event_info_map = df[['event_info', 'event_info_id']].drop_duplicates()

    # The implicit library expects data as a item-user matrix so we
    # create two matricies, one for fitting the model (item-user)
    # and one for recommendations (user-item)

    # relevance = 'normalized_views' if normalized else 'views'
    relevance = 'views'
    sparse_item_user = sparse.csr_matrix((df[relevance].astype(float),
                                         (df['event_info_id'],  df['seq_id'])))
    sparse_user_item = sparse.csr_matrix((df[relevance].astype(float),
                                         (df['seq_id'], df['event_info_id'])))

    # Initialize the als model and fit it using the sparse item-user matrix
    model = implicit.als.AlternatingLeastSquares(factors = factors,
                                                 regularization = regularization,
                                                 iterations = iterations)

    # Calculate the confidence by multiplying it by our alpha value.
    sparse_item_user_conf = (sparse_item_user * alpha_val)#.astype('double')

    #Fit the model
    model.fit(sparse_item_user_conf)

    return model, seq_map, event_info_map, sparse_user_item
