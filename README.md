Tomt project with ColBERT

Assume there is a "movie" folder in "dataset" folder, which is a movie folder which is created by downloading reddit tomt dataset.
The ColBERT directory is downloaded from v1 branch of official colbert implementation, which can be found in https://github.com/stanford-futuredata/ColBERT.
The modeling directory in it contains ColBERT model. The functions for in-batch negative score calculation and weighted score calculation are added.

Data.py file contains files that preprocess the data. 
infrence.py file contains functions that calculates embeddings for all documents given a model.
document.pickle contains a dict where key is a document key in the original dataset, and value is a tuple of document in original data and first paragraph of corresponding wikipedia article(or movie name if the article is not found). It is generated using Wikipedia library, and code to make this file is in tomt.ipynb

We conducted experiments on three settings and calculated the recall-10 rate on validation data: Vanilla colbert, Weighted colbert, weighted colbert + additional data.
We found that using Colbert gives much better results than DPR, but using weighted colbert or additional data didn't change the result a lot.
