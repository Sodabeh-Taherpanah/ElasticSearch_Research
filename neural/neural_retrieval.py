from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import json
import numpy as np
from typing import List, Dict, Any
import urllib3
import matplotlib.pyplot as plt
from collections import defaultdict

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "MyElasticPass123"),
    verify_certs=False,
    ssl_show_warn=False,
    request_timeout=120
)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Neural models
e5_model = SentenceTransformer("intfloat/e5-large-v2", device=device)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=str(device))

def enhance_biomedical_query(query: str) -> str:
    """Neural query enhancement - no traditional expansion"""
    # Pure neural approach , the model handle semantics
    return f"query: {query} biomedical research clinical study"

def get_neural_embedding(text: str) -> np.ndarray:
    """Get high-quality neural embedding"""
    return e5_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def pure_neural_search(query: str, index_name: str = "neural_biomedical_index", top_k: int = 50) -> List[Dict]:
    """Pure neural retrieval without BM25"""
    
    enhanced_query = enhance_biomedical_query(query)
    print(f"🧠 Neural query: {enhanced_query}")
    
    query_vector = get_neural_embedding(enhanced_query).tolist()
    
    try:
        # Pure vector search 
        search_body = {
            "knn": {
                "field": "dense_vector",
                "query_vector": query_vector,
                "k": 100,
                "num_candidates": 300
            },
            "_source": ["title", "text", "_id"],
            "size": 100
        }
        
        response = es.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        if not hits:
            return []
        
        # Neural re-ranking with cross-encoder
        texts_to_rerank = [hit['_source']['text'][:1000] for hit in hits]
        rerank_scores = cross_encoder.predict([(enhanced_query, text) for text in texts_to_rerank])
        
        # Apply neural re-ranking
        for i, hit in enumerate(hits):
            hit['_neural_score'] = float(rerank_scores[i])
            hit['_final_score'] = (hit['_score'] * 0.1) + (rerank_scores[i] * 1)
        
        # Sort by neural score
        hits.sort(key=lambda x: x['_final_score'], reverse=True)
        
        return hits[:top_k]
        
    except Exception as e:
        print(f"-- Neural search failed: {e}")
        return []

def evaluate_neural_retrieval(prof_dataset_path: str = "final_dataset.jsonl") -> Dict[str, Any]:
    """Comprehensive evaluation of pure neural retrieval"""
    
    try:
        with open(prof_dataset_path, 'r') as f:
            prof_data = json.loads(f.readline().strip())
        
        query = prof_data["question"]
        expected_docs = set(prof_data["expected_resources"])
        
        print(f"-- Evaluating: '{query}'")
        print(f"-- Expected documents: {len(expected_docs)}")
        
        # Pure neural retrieval
        results = pure_neural_search(query, top_k=50)
        retrieved_docs = [hit['_id'] for hit in results]
        relevant_retrieved = [doc for doc in retrieved_docs if doc in expected_docs]
        missing_docs = expected_docs - set(retrieved_docs)
        
        # Calculate metrics
        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
        recall = len(relevant_retrieved) / len(expected_docs) if expected_docs else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Detailed analysis
        print(f"\n -- NEURAL RETRIEVAL RESULTS:--")
        print(f" --Precision: {precision:.2%} ({len(relevant_retrieved)}/{len(retrieved_docs)})")
        print(f" --Recall: {recall:.2%} ({len(relevant_retrieved)}/{len(expected_docs)})")
        print(f"✅-- F1-Score: {f1_score:.2%}")
        
        if relevant_retrieved:
            print(f"✅ Relevant found: {sorted(relevant_retrieved)}")
        
        print(f"❌ Missing documents: {len(missing_docs)}")
        if missing_docs:
            print("   " + "\n   ".join(sorted(missing_docs)))
        
        # Top results analysis
        print(f"\n🏆 TOP NEURAL RESULTS:")
        for i, hit in enumerate(results[:10]):
            relevance = "✓" if hit['_id'] in expected_docs else "✗"
            print(f"  {i+1}. [{relevance}] {hit['_id']} - Neural: {hit['_final_score']:.3f}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "relevant_retrieved": relevant_retrieved,
            "missing_docs": list(missing_docs),
            "retrieved_count": len(retrieved_docs),
            "expected_count": len(expected_docs)
        }
        
    except Exception as e:
        print(f"-- Evaluation failed: {e}")
        return {}


def evaluate_neural_retrieval_all(prof_dataset_path: str = "final_dataset.jsonl") -> Dict[str, Any]:
    """Evaluate pure neural retrieval for whole queries in the dataset."""
    
    total_relevant_retrieved = 0
    total_retrieved = 0
    total_expected = 0
    all_missing_docs = []
    
    try:
        with open(prof_dataset_path, 'r') as f:
            for line in f:
                prof_data = json.loads(line.strip())
                
                query = prof_data["question"]
                expected_docs = set(prof_data["expected_resources"])
                
               
                results = pure_neural_search(query, top_k=50)
                retrieved_docs = [hit['_id'] for hit in results]
                
                relevant_retrieved = [doc for doc in retrieved_docs if doc in expected_docs]
                missing_docs = expected_docs - set(retrieved_docs)
                
                # Accumulate totals
                total_relevant_retrieved += len(relevant_retrieved)
                total_retrieved += len(retrieved_docs)
                total_expected += len(expected_docs)
                all_missing_docs.extend(list(missing_docs))
                
    except Exception as e:
        print(f" Evaluation failed: {e}")
        return {}
    
    # Compute overall metrics
    precision = total_relevant_retrieved / total_retrieved if total_retrieved else 0
    recall = total_relevant_retrieved / total_expected if total_expected else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n NEURAL RETRIEVAL OVERALL RESULTS:")
    print(f" Precision: {precision:.2%} ({total_relevant_retrieved}/{total_retrieved})")
    print(f"-- Recall: {recall:.2%} ({total_relevant_retrieved}/{total_expected})")
    print(f"-- F1-Score: {f1_score:.2%}")
    print(f" -- Total missing documents: {len(all_missing_docs)}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_retrieved": total_retrieved,
        "total_expected": total_expected,
        "total_relevant_retrieved": total_relevant_retrieved,
        "missing_docs": all_missing_docs
    }

def generate_evaluation_report(results: Dict[str, Any]):
    """Generate comprehensive evaluation report"""
    
    print("\n" + "="*60)
    print("-- NEURAL RETRIEVAL EVALUATION REPORT")
    print("="*60)
    
    print(f"-- Final Score: {results['f1_score']:.2%} F1")
    print(f"-- Precision: {results['precision']:.2%}")
    print(f"-- Recall: {results['recall']:.2%}")
    print(f"-- Documents retrieved: {results['retrieved_count']}")
    print(f"-- Relevant found: {len(results['relevant_retrieved'])}/{results['expected_count']}")
    print(f"-- Documents missed: {len(results['missing_docs'])}")
    

if __name__ == "__main__":
    print("-- Neural Retrieval Evaluation --")
    print("="*60)
    
    results = evaluate_neural_retrieval("final_dataset.jsonl")

    if results:
        generate_evaluation_report(results)
        
        # Save results
        with open("neural_retrieval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
    
    print("="*60)
    