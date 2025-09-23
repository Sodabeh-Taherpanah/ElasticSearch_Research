from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import torch
import json
from tqdm import tqdm
import urllib3
import numpy as np

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize Elasticsearch
es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "MyElasticPass123"),
    verify_certs=False,
    ssl_show_warn=False,
    request_timeout=120
)

# Initialize E5-large model (best for pure neural retrieval)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
e5_model_name = "intfloat/e5-large-v2"  # 1024 dimensions for best performance
e5_model = SentenceTransformer(e5_model_name, device=device)
print(f"🧠 Using {e5_model_name} on {device}")

def get_dense_embedding(text):
    """Generate high-quality embeddings"""
    return e5_model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()

def create_neural_index(index_name="neural_biomedical_index"):
    """Create index optimized for pure neural retrieval"""
    
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print("🗑️ Deleted existing index")
    
    # Pure neural index - no text analysis needed for BM25
    index_body = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},  # Only for display
                "text": {"type": "text"},   # Only for display
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": e5_model.get_sentence_embedding_dimension(),  # 1024 for large-v2
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=index_body)
    print(f"✅ Created pure neural index: {index_name}")
    return index_name

def index_documents_neural(corpus_path="corpus.jsonl", index_name="neural_biomedical_index"):
    """Index documents using only neural embeddings"""
    
    index_name = create_neural_index(index_name)
    
    successful_count = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="🧠 Neural indexing")):
            try:
                doc = json.loads(line.strip())
                
                # Generate high-quality embedding
                embedding = get_dense_embedding(doc["text"])
                
                es_doc = {
                    "title": doc.get("title", ""),
                    "text": doc["text"],
                    "dense_vector": embedding
                }
                
                es.index(
                    index=index_name, 
                    id=doc["_id"], 
                    document=es_doc,
                    timeout="120s"
                )
                successful_count += 1
                
            except Exception as e:
                print(f"❌ Error indexing {doc.get('_id', 'unknown')}: {e}")
                continue

    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)["count"]
    print(f"✅ Neural indexing complete! Documents: {count}")
    
    # Verify dimensions
    verify_indexing(index_name)

def verify_indexing(index_name):
    """Verify neural indexing was successful"""
    try:
        sample = es.search(index=index_name, body={"size": 1, "_source": ["dense_vector"]})
        if sample['hits']['hits']:
            dims = len(sample['hits']['hits'][0]['_source']['dense_vector'])
            expected = e5_model.get_sentence_embedding_dimension()
            print(f"🔍 Vector dimensions: {dims} (expected: {expected})")
            return dims == expected
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    index_documents_neural("corpus.jsonl", "neural_biomedical_index")