import json
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MeaningfulSimpleRAG:
    def __init__(self):
        self.documents = []
        self.doc_ids = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def load_corpus(self, corpus_path="corpus.jsonl"):
        """Load documents and create meaningful TF-IDF vectors"""
        print("📦 Loading documents and building TF-IDF...")
        
        texts = []
        with open(corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line.strip())
                self.documents.append(doc['text'])
                self.doc_ids.append(doc['_id'])
                texts.append(doc['text'])
        
        # Build meaningful TF-IDF vectors
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"✅ Loaded {len(self.documents)} documents with TF-IDF")
            
    def retrieve(self, query, top_k=50):
        """Meaningful TF-IDF similarity search"""
        # Convert query to TF-IDF
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.doc_ids[idx],
                'score': similarities[idx],
                'text': self.documents[idx][:200] + "..." if len(self.documents[idx]) > 200 else self.documents[idx]
            })
            
        return results

    def enhance_query(self, query):
        """Simple query enhancement for better TF-IDF matching"""
        # Basic biomedical term expansion
        expansion_terms = {
            'statin': 'statin cholesterol drug hmg coa reductase',
            'breast cancer': 'breast cancer mammary carcinoma tumor',
            'cause': 'cause effect risk factor association'
        }
        
        enhanced = query.lower()
        for term, expansion in expansion_terms.items():
            if term in enhanced:
                enhanced += " " + expansion
                
        return enhanced

def evaluate_meaningful_rag():
    """Evaluate meaningful TF-IDF RAG baseline"""
    
    rag = MeaningfulSimpleRAG()
    rag.load_corpus()
    
    with open("professor_dataset.jsonl", 'r') as f:
        prof_data = json.loads(f.readline().strip())
    
    query = prof_data["question"]
    expected_docs = set(prof_data["expected_resources"])
    
    # Enhance query for better matching
    enhanced_query = rag.enhance_query(query)
    print(f"🔍 Evaluating TF-IDF RAG: '{query}'")
    print(f"📋 Enhanced query: '{enhanced_query}'")
    print(f"📋 Expected documents: {len(expected_docs)}")
    
    results = rag.retrieve(enhanced_query, top_k=50)
    retrieved_docs = [hit['id'] for hit in results]
    relevant_retrieved = [doc for doc in retrieved_docs if doc in expected_docs]
    
    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(relevant_retrieved) / len(expected_docs) if expected_docs else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 TF-IDF RAG Results:")
    print(f"✅ Precision: {precision:.2%} ({len(relevant_retrieved)}/{len(retrieved_docs)})")
    print(f"✅ Recall: {recall:.2%} ({len(relevant_retrieved)}/{len(expected_docs)})")
    print(f"✅ F1-Score: {f1_score:.2%}")
    
    # Show some results
    print(f"\n🔍 Retrieved {len(retrieved_docs)} documents")
    print(f"✅ Relevant found: {len(relevant_retrieved)}")
    if relevant_retrieved:
        print(f"📝 Relevant documents: {relevant_retrieved[:10]}")  # First 10 only
    
    return precision, recall, f1_score

if __name__ == "__main__":
    evaluate_meaningful_rag()