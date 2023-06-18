import torch
import data
from colbert.modeling.inference import ModelInference
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
query_maxlen = 512
doc_maxlen = 512
def get_document_embeddings(model,documents=None):
  model.eval()
  modelInference = ModelInference(model)
  if documents ==None:
   documents = data.load_documents()
  embeddings = dict()
  tokenizer = DocTokenizer(doc_maxlen=doc_maxlen)
  i = 0
  for docId in documents:
    
    if(i % 100 == 0):print(i)
    i+=1
    document = documents[docId]
    if type(document) == str:
     embeddings[docId] = modelInference.doc(*tokenizer.tensorize([document]))
    else:
      embeddings[docId] = (modelInference.doc(*tokenizer.tensorize([document[0]])),modelInference.doc(*tokenizer.tensorize([document[1]])))
  return embeddings
def get_query_ranking(model,query,docId,embeddings):
  model.eval()
  modelInference = ModelInference(model)
  tokenizer = QueryTokenizer(query_maxlen)
  Q = modelInference.query(*tokenizer.tensorize([query]))
  score = modelInference.score(Q,embeddings[docId])
  rank = 1
  for doc in embeddings:
    if doc == docId:continue
    if modelInference.score(Q,embeddings[doc])>score:rank+=1
  return rank
def recall(model,split,embeddings,num):
    dataset = data.make_dataset(split)


  