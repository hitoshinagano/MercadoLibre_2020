MercadoLibre_competicao
==============================

Code for Mercado Libre Data Challenge 2020.


# solution

The solution is based on two approaches:
1. matrix factorization, for seq's that had at least one view
2. similarity via sentence embeddings from search and item descriptions, for seq's that had no views

I have used the term `seq` (sequence) to denote user, throughout the code and here.

# notebooks

    ├── notebooks
    │   ├── pred_base                     <- baseline submissions that will get you around ndcg=0.22... 
    │   │                                    in public leaderboard.
    │   ├── EDA_dataprep                  <- summarizes some finds in EDA and prepares data for modeling.
    │   ├── dataprep_customembedding      <- word vectors using gensim.
    │   ├── implicit                      <- grid search for validation with implicit. It also contains 
    │   │                                    a prediction script to produce submissions.
    │   ├── SEARCH_dataprep               <- embeddings for the search texts, from custom word vectors 
    │   │                                    or from pre-trained word vectors.
    │   └── SEARCH_validation_prediction  <- recommendations based on similarity of search texts 
                                             (queries) and the description of the item.
    
# src

    ├── src
    │   │
    │   ├── features       
    │   │   └── utils.py          <- various functions to read, write, 
    │   │                            sample and also to clean, transform dataset
    │   │
    │   ├── models         
    │   │   │          
    │   │   ├── predict_model.py  <- predict with implicit model
            └── train_model.py    <- fit implicit model
    

# results

NDCG results:
* 0.27 in public
* 0.26 in private

16th/180 place in both rankings.



