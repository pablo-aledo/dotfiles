# import AquilaDB client
from aquiladb import AquilaClient as acl

# create DB instance
db = acl('localhost', 50051)

# convert a sample document
# convertDocument
sample = db.convertDocument([0.1,0.2,0.3,0.4], {"hello": "world"})

# add document to AquilaDB
db.addDocuments([sample])

# note that, depending on your default configuration, 
# you need to add docs.vecount number of documents 
# before k-NN search

# create a k-NN search vector
vector = db.convertMatrix([0.1,0.2,0.3,0.4])

# perform k-NN from AquilaDB
k = 10
result = db.getNearest(vector, k)
result
