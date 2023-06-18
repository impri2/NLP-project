import json
import torch
import wikipedia
DATASET_PATH = '/content/drive/MyDrive/dataset/Movies/'
def load_queries():#return query id to text dict
  query_file = open(DATASET_PATH+'queries.json')
  lines = query_file.readlines()
  queries=dict()
  for line in lines:
    query = json.loads(line)
    queries[query["id"]] = query['title'] + ' ' + query['description']
  query_file.close()
  return queries
def load_documents():#return document id to text dict
  documents=dict()
  for filename in ['documents.json','hard_negative_documents.json','negative_documents.json']:

   document_file = open(DATASET_PATH+filename)
   lines = document_file.readlines()
   for line in lines:
    document = json.loads(line)
    documents[document["id"]] = document['text']
   document_file.close()
  return documents
def save_documents_with_summaries():
  i = 0
  documents = dict()
  for filename in ['documents.json','hard_negative_documents.json','negative_documents.json']:
    document_file = open(DATASET_PATH+filename)
    lines = document_file.readlines()
    for line in lines:
      document = json.loads(line)
      plot = document[ 'text']
      title = document['title']
      summary = ""
      try:
        summary =  wikipedia.summary(title)
        i += 1
        print(i)
      except:
        summary = title
        print("not found")
      documents[document['id']] = (plot,summary)
  return documents
      
def load_bm25_negative():
  bm25_file = open(DATASET_PATH+'bm25_hard_negatives_all.json')
  jsonval = json.loads(bm25_file.read())
  bm25_file.close()
  return jsonval

def make_dataset(split,documents=None):
  assert split == 'train' or split == 'validation' or split == 'test'
  dataset_file = open(DATASET_PATH + 'splits/' + split + '/qrels.txt')
  queries = load_queries()
  negatives = load_bm25_negative()
  if documents == None:
    documents = load_documents()
  dataset = []
  for line in dataset_file.readlines():
    line = line.split()
    try:
     dataset.append({'query':queries[line[0]],'document':documents[line[2]]
     ,'negative':documents[negatives[line[0]][0]]})
    except:# sometimes a query doesn't have its text. Maybe the post is removed?
      pass
  dataset_file.close()
  return dataset
  
class TomtDataset(torch.utils.data.Dataset):

  def __init__(self,documents=None):
    self.dataset = make_dataset('train',documents)
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self,i):
    return self.dataset[i]
class TomtValidDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.dataset = make_dataset('validation')
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self,i):
    return self.dataset[i]



