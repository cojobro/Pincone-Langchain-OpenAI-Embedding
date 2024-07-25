import os
import re
import time
import pdfplumber
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize OpenAI
client = OpenAI(
    api_key="openai-api-key"
)  # get API key from platform.openai.com


MODEL = "text-embedding-3-small"

res = client.embeddings.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], model=MODEL
)

# we can extract embeddings to a list
embeds = [record.embedding for record in res.data]
len(embeds)

# Initialize Pinecone
pc = Pinecone(api_key="pinecone-api-key")  # get API key from pinecone.io

spec = ServerlessSpec(cloud="aws", region="us-east-1")

# Define the index name
index_name = "my-index"

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=len(embeds[0]),  # dimensionality of text-embed-3-small
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Instantiate the index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text

def process_pdf(file_path):
    # create a loader
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [str(doc) for doc in documents]
    return texts


file_path = "relative file path"  # Replace with your actual file path like "./docs/sample.pdf"
texts = process_pdf(file_path)

from tqdm.auto import tqdm

count = 0  # we'll use the count to create unique IDs
batch_size = 32  # process everything in batches of 32

for i in tqdm(range(0, len(texts), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(texts))
    # get batch of lines and IDs
    lines_batch = texts[i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = client.embeddings.create(input=lines_batch, model=MODEL)
    embeds = [record.embedding for record in res.data]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))
