import pandas as pd
import numpy as np
import os
from sklearn.metrics import ndcg_score

def predict_implicit_model(model, sparse_user_item,
                           seq_map, event_info_map,
                           test_shifted_seq_vals,
                           N, test_offset,
                           validation = True, true_df = None, item_domain = None):

    """
    Args:
    ====
    * model, sparse_user_item
    * seq_map, event_info_map, test_shifted_seq_vals
    * N: quantity of recomended items
    * test_offset
    * validation, true_df, item_domain
    """

    recs = seq_map[seq_map.seq.isin(test_shifted_seq_vals)].copy()

    # whole test takes about 3 hours: 100%|██████████| 165700/165700 [3:08:36<00:00, 14.64it/s]
    recommender = lambda x: model.recommend(x, sparse_user_item,
                                            filter_already_liked_items = False, N = N)
    recs['implicit_preds'] = recs.seq_id.progress_apply(recommender)
    recs['event_info_id'] = recs.implicit_preds.apply(lambda x: [y[0] for y in x])
    recs['top_score'] = recs.implicit_preds.apply(lambda x: x[0][1])

    # recs.top_score.hist(bins = 30)

    recs = pd.merge(recs.explode('event_info_id'), event_info_map, how = 'left')

    qty_seqs = recs.shape[0] // N
    recs['pred'] = list(range(N)) * qty_seqs

    top_score = recs[['seq', 'top_score']].groupby('seq').top_score.first().dropna()

    pred = recs.pivot(index = 'seq', columns = 'pred', values = 'event_info')
    pred.sort_index(inplace = True)
    pred = pd.concat([pred, top_score], axis = 1)
    pred.index = [idx - test_offset for idx in pred.index]

    if validation:

        pred.rename(columns = {c: str(c) for c in range(N)}, inplace = True)

        pred.index.name = 'seq'
        pred = pred.reset_index()

        pred = pd.merge(pred, true_df, how = 'left')

        pred = pd.merge(pred, item_domain, how = 'left',
                        left_on = 'item_bought', right_on = 'item_id')
        pred.rename(columns = {'domain_id': 'item_bought_domain'}, inplace = True)
        pred.drop('item_id', axis = 1, inplace = True)

        for c in range(N):
            pred = pd.merge(pred, item_domain, how = 'left',
                            left_on = str(c), right_on = 'item_id')
            pred.rename(columns = {'domain_id': 'domain_id_' + str(c)}, inplace = True)
            pred.drop('item_id', axis = 1, inplace = True)

        for c in range(N):
            pred['rel_item_' + str(c)]   = (pred[str(c)] == pred.item_bought) * 12
            pred['rel_domain_' + str(c)] = (pred['domain_id_' + str(c)] ==
                                            pred.item_bought_domain).astype(int)

        pred['relevances_item'] = pred.filter(like = 'rel_item').apply(list, axis = 1)
        pred['relevances_domain'] = pred.filter(like = 'rel_domain').apply(list, axis = 1)

        sum_relevances = lambda x: np.array(x.relevances_item) + np.array(x.relevances_domain)
        pred['relevances'] = pred[['relevances_item', 'relevances_domain']].apply(sum_relevances,
                                                                                  axis = 1)

        N_to_1 = [list(range(N, 0, -1))]

        pred['ndcg'] = pred.relevances.apply(lambda x: ndcg_score([x], N_to_1))

        # print('mean ndcg:', pred.ndcg.mean())
        # print('proportion of ndcg higher than zero:', (pred.ndcg > 0).mean())

        return pred

    else:
        return pred


def score_pred(pred, true_df, item_domain, N = None):

    if not N: N = pred.shape[1]

    pred.rename(columns = {c: str(c) for c in range(N)}, inplace = True)

    pred.index.name = 'seq'
    pred = pred.reset_index()

    pred = pd.merge(pred, true_df, how = 'left')

    pred = pd.merge(pred, item_domain, how = 'left',
                    left_on = 'item_bought', right_on = 'item_id')
    pred.rename(columns = {'domain_id': 'item_bought_domain'}, inplace = True)
    pred.drop('item_id', axis = 1, inplace = True)

    for c in range(N):
        pred = pd.merge(pred, item_domain, how = 'left',
                        left_on = str(c), right_on = 'item_id')
        pred.rename(columns = {'domain_id': 'domain_id_' + str(c)}, inplace = True)
        pred.drop('item_id', axis = 1, inplace = True)

    for c in range(N):
        pred['rel_item_' + str(c)]   = (pred[str(c)] == pred.item_bought) * 12
        pred['rel_domain_' + str(c)] = (pred['domain_id_' + str(c)] ==
                                        pred.item_bought_domain).astype(int)

    pred['relevances_item'] = pred.filter(like = 'rel_item').apply(list, axis = 1)
    pred['relevances_domain'] = pred.filter(like = 'rel_domain').apply(list, axis = 1)

    sum_relevances = lambda x: np.array(x.relevances_item) + np.array(x.relevances_domain)
    pred['relevances'] = pred[['relevances_item', 'relevances_domain']].apply(sum_relevances,
                                                                              axis = 1)

    N_to_1 = [list(range(N, 0, -1))]

    pred['ndcg'] = pred.relevances.apply(lambda x: ndcg_score([x], N_to_1))

    return pred
