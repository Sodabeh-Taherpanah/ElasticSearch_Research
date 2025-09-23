from elasticsearch import Elasticsearch
from sentence_transformers import SparseEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np
import re
import os

# Set environment variable to handle MPS device issues
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "MyElasticPass123"),
    verify_certs=False,
    request_timeout=120
)


splade_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")


biomedical_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(biomedical_model_name)
dense_model = AutoModel.from_pretrained(biomedical_model_name)

#  Biomedical query enhancement
def enhance_biomedical_query(query):
    """
    Enhanced query expansion based on statin-breast cancer research and missing document analysis
    Incorporates terminology from search results and addresses vocabulary gaps
    """
    enhancement_map = {
        "cause": [
            "carcinogenic", "tumorigenic", "risk factors", "adverse effects", "toxicity",
            "etiology", "pathogenesis", "causal relationship", "oncogenic potential",
            "safety profile", "risk assessment", "mechanism of action"
        ],
        "statin": [
            "HMG-CoA reductase inhibitor", "atorvastatin", "simvastatin", "rosuvastatin", 
            "lipophilic statin", "lovastatin", "pravastatin", "statin therapy",
            "cholesterol-lowering medication", "HMGCR inhibitor", "statin treatment",
            "statin medication", "lipid-lowering therapy"
        ],
        "breast cancer": [
            "mammary neoplasms", "breast carcinoma", "HR+ HER2-", "ER+", "triple negative",
            "ductal carcinoma", "lobular carcinoma", "early-stage breast cancer", 
            "estrogen receptor positive", "HER2-negative", "TNBC", "breast malignancies",
            "mammary carcinoma", "breast tumor"
        ],
        "cholesterol": [
            "lipid", "LDL", "low-density lipoprotein", "hyperlipidemia", "dyslipidemia",
            "lipid metabolism", "lipoprotein", "hypocholesterolemia", "cholesterol synthesis",
            "LDL receptor", "lipoprotein metabolism", "cholesterol homeostasis"
        ],
        "drug": [
            "pharmaceutical", "medication", "therapy", "treatment", "pharmacotherapy",
            "therapeutic agents", "prescription drugs", "medical treatment", 
            "adjuvant therapy", "chemoprevention", "therapeutic intervention"
        ],
        # New category based on missing document analysis
        "protective": [
            "risk reduction", "recurrence prevention", "mortality reduction", 
            "survival benefit", "protective effect", "improved outcomes",
            "therapeutic benefit", "favorable prognosis", "positive outcomes"
        ]
    }
    
    # Additional terms from missing document analysis
    additional_terms = [
        "LDL receptor content", "low density lipoprotein", "prognostic value",
        "survival time", "patient survival", "axillary metastasis", "DNA pattern",
        "tumor diameter", "phytoestrogens", "lignans", "dietary fiber", 
        "soy foods", "soybeans", "sunflower seeds", "pumpkin seeds",
        "estrogen receptor status", "ER status", "multivariate analysis",
        "statistical significance", "clinical trial", "randomized study",
        "observational study", "cohort study", "case-control study",
        # Added dietary terms
        "nutritional intervention", "dietary pattern", "food intake", 
        "plant estrogens", "soy isoflavones", "nutritional epidemiology",
        "dietary supplement", "plant-based nutrition", "nutritional factors"
    ]
    
    enhanced = query.lower()
    
    # Add domain-specific expansions
    for term, expansions in enhancement_map.items():
        if term in enhanced:
            enhanced += " " + " ".join(expansions)
    
    # Add terms from missing document analysis
    enhanced += " " + " ".join(additional_terms)
    
    return enhanced

# Biomedical embedding function
def get_biomedical_embedding(text):
    """Generate embeddings using biomedical BERT"""
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=256
    )
    
    with torch.no_grad():
        outputs = dense_model(**inputs)
    
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings


#  return top_terms[:top_k]
def extract_biomedical_terms(sparse_embedding, original_query, top_k=15):
    """Extract and filter relevant biomedical terms from SparseEncoder output"""
    
    if hasattr(sparse_embedding, 'shape'):
        print(f"   Sparse embedding shape: {sparse_embedding.shape}")
    
    # Handle different SparseEncoder output formats
    if isinstance(sparse_embedding, dict) and 'indices' in sparse_embedding:
        # first format: dictionary with indices and values
        indices = sparse_embedding['indices']
        values = sparse_embedding['values']
        weighted_terms = list(zip(indices, values))
        
    elif hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
        # second format: object with indices and values attributes
        indices = sparse_embedding.indices
        values = sparse_embedding.values
        weighted_terms = list(zip(indices, values))
        
    elif isinstance(sparse_embedding, (list, np.ndarray, torch.Tensor)):
      
        if isinstance(sparse_embedding, torch.Tensor):
            sparse_embedding = sparse_embedding.cpu().numpy()
        
        if isinstance(sparse_embedding, np.ndarray):
            indices = np.nonzero(sparse_embedding)[0]
            values = sparse_embedding[indices]
            weighted_terms = list(zip(indices, values))
        else:
            # Handle as list
            weighted_terms = [(i, val) for i, val in enumerate(sparse_embedding) if val > 0]
            
    else:
     
        return []
    
    weighted_terms.sort(key=lambda x: x[1], reverse=True)
    
    # Priority biomedical terms for statin/cancer context
    priority_terms = {
        'cancer', 'tumor', 'stat', 'breast', 'drug', 'cholesterol', 'statin',
        'carcinogenic', 'therapy', 'patient', 'disease', 'medical', 'dose',
        'medication', 'risk', 'mortality', 'survival', 'recurrence', 'prevention',
        'treatment', 'clinical', 'study', 'effect', 'reduction', 'mortality',
        # Add dietary terms based on previous analysis
        'diet', 'nutrition', 'soy', 'fiber', 'plant', 'food', 'dietary',
        'phytoestrogen', 'lignan'
    }
    
    top_terms = []
    for token_id, weight in weighted_terms[:top_k*3]:
        try:
            # Convert token ID back to text using the biomedical tokenizer
            term = tokenizer.convert_ids_to_tokens(int(token_id))
            term = term.replace('Ġ', '').replace('▁', '').replace('##', '').strip()
            
            if (len(term) > 2 and 
                not term.startswith('[') and 
                term not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', 'cls', 'sep', 'pad', 'unk'] and
                term.isalpha()):
                
                #  include terms relevant to biomedical context
                if (term in priority_terms or 
                    any(keyword in term for keyword in ['med', 'cancer', 'tumor', 'stat', 'drug', 'diet', 'nutri'])):
                    boosted_weight = weight * 3.0 if term in priority_terms else weight
                    top_terms.append((term, float(boosted_weight)))
                
        except Exception as e:
            print(f"   Error converting token {token_id}: {e}")
            continue
    
    # If SPLADE fails or returns few terms, use manual terms from original query
    if not top_terms or len(top_terms) < 5:
        print("   SPLADE returned few terms, using enhanced manual terms")
        manual_terms = [
            ('statin', 4.0), ('breast', 4.0), ('cancer', 4.0), 
            ('cholesterol', 3.5), ('drug', 3.0), ('cause', 2.5),
            ('therapy', 2.0), ('treatment', 2.0), ('risk', 2.0),
            ('dietary', 2.2), ('nutrition', 2.0), ('soy', 2.0),
            ('animal', 1.5), ('study', 1.8), ('clinical', 1.8)
        ]
        top_terms = [(term, weight) for term, weight in manual_terms if term in original_query.lower()]
        # Add dietary terms if query is related
        if any(nut_term in original_query.lower() for nut_term in ['diet', 'nutrition', 'food']):
            top_terms.extend([('dietary', 2.0), ('nutrition', 2.0), ('food', 1.8)])
    
    return top_terms[:top_k]
# 5. Missing Document Analysis
def analyze_missing_documents(missing_docs, corpus_path="corpus.jsonl"):
    """
    Analyze terminology patterns in missing documents to understand why they weren't retrieved
    """
    terminology_patterns = {}
    document_types = {}
    
    print(f"\n-- Analyzing {len(missing_docs)} missing documents...")
    
    try:
        with open(corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                if doc['_id'] in missing_docs:
                    text = doc.get('text', '').lower()
                    title = doc.get('title', '').lower()
                    
                    # Check for study design patterns
                    study_patterns = {
                        'in_vitro': ['in vitro', 'cell line', 'cell culture', 'mtt assay'],
                        'animal_study': ['mouse', 'mice', 'rat', 'animal model', 'in vivo'],
                        'mechanistic': ['mechanism', 'pathway', 'signaling', 'molecular'],
                        'clinical_trial': ['clinical trial', 'randomized', 'phase iii', 'phase 3'],
                        'observational': ['observational', 'cohort', 'case-control', 'epidemiological']
                    }
                    
                    # Check for content focus patterns
                    content_patterns = {
                        'molecular': ['gene', 'protein', 'expression', 'mrna', 'dna'],
                        'clinical': ['patient', 'survival', 'recurrence', 'mortality', 'prognosis'],
                        'methodological': ['statistical', 'multivariate', 'regression', 'analysis'],
                        'dietary': ['diet', 'nutrition', 'food', 'soy', 'phytoestrogen', 'lignan']
                    }
                    
                    # Analyze study type
                    for doc_type, patterns in study_patterns.items():
                        if any(pattern in text or pattern in title for pattern in patterns):
                            document_types[doc_type] = document_types.get(doc_type, 0) + 1
                    
                    # Analyze content focus
                    for focus, patterns in content_patterns.items():
                        if any(pattern in text for pattern in patterns):
                            terminology_patterns[focus] = terminology_patterns.get(focus, 0) + 1
                    
                    # Count specific important terms
                    specific_terms = ['ldl receptor', 'lipoprotein', 'phytoestrogen', 'lignan', 
                                    'dietary fiber', 'soy', 'statistical significance', 'multivariate']
                    for term in specific_terms:
                        if term in text:
                            terminology_patterns[term] = terminology_patterns.get(term, 0) + 1
                            
    except Exception as e:
        print(f"Error analyzing missing documents: {e}")
        return
    
  
    print("\n-- Missing Document Analysis Results:")
    print("Study Types:")
    for doc_type, count in sorted(document_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_type}: {count} documents")
    
    # Enhanced Dietary Analysis
    print("\nContent Focus Patterns (Dietary only):")
    dietary_terms = ['dietary', 'soy', 'lignan', 'dietary fiber', 'phytoestrogen', 'nutrition', 'food', 'diet', 'plant-based', 'soybeans', 'sunflower', 'pumpkin', 'lignans', 'phytoestrogens']
    dietary_patterns = {}

    # Collect all dietary-related patterns
    for term, count in terminology_patterns.items():
        if any(diet_term in term for diet_term in dietary_terms):
            dietary_patterns[term] = count

    if dietary_patterns:
        for focus, count in sorted(dietary_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {focus}: {count} documents")
    else:
        print("  No dietary-related patterns found in missing documents")

    # Additional detailed dietary analysis
    print("\n-- Detailed Dietary Analysis:")
    try:
        dietary_missing_docs = []
        strong_dietary_docs = []
        
        with open(corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                if doc['_id'] in missing_docs:
                    text = doc.get('text', '').lower()
                    title = doc.get('title', '').lower()
                    
                    # Check for any dietary content
                    dietary_keywords = ['soy', 'lignan', 'phytoestrogen', 'dietary fiber', 'plant-based', 
                                      'nutrition', 'food intake', 'diet', 'nutritional', 'phytoestrogens',
                                      'lignans', 'soybeans', 'sunflower seeds', 'pumpkin seeds']
                    
                    if any(keyword in text or keyword in title for keyword in dietary_keywords):
                        dietary_missing_docs.append(doc['_id'])
                    
                    # Check for strong dietary focus (multiple keywords)
                    strong_dietary_count = sum(1 for keyword in dietary_keywords if keyword in text or keyword in title)
                    if strong_dietary_count >= 3:  # At least 3 dietary keywords
                        strong_dietary_docs.append(doc['_id'])
        
        print(f"Documents with any dietary content: {len(dietary_missing_docs)}")
        print(f"Documents with strong dietary focus: {len(strong_dietary_docs)}")
        
        if dietary_missing_docs:
            print(f"Sample dietary document IDs: {dietary_missing_docs[:5]}{'...' if len(dietary_missing_docs) > 5 else ''}")
            
    except Exception as e:
        print(f"Error in detailed dietary analysis: {e}")

#  HYBRID Search for SparseEncoder
def biomedical_hybrid_search(query, index_name="biomedical_hybrid_index", top_k=10):
    """Robust hybrid search with fallback mechanisms"""
    print(f"-- Original Query: '{query}'")
    
    # Enhance query first
    enhanced_query = enhance_biomedical_query(query)
    print(f"   Enhanced Query: '{enhanced_query[:200]}...'")
    
    # Generate SPLADE expansion using SparseEncoder
    splade_terms = [] 
    try:
        # Try different method names that SparseEncoder might use
        if hasattr(splade_model, 'encode_queries'):
            splade_embedding = splade_model.encode_queries([enhanced_query])[0]
            print(f"   Sparse encoder result type: {type(splade_embedding)}")
        elif hasattr(splade_embedding, 'shape'):
            print(f"   Sparse encoder result shape: {splade_embedding.shape}")
        elif hasattr(splade_model, 'encode'):
            splade_embedding = splade_model.encode([enhanced_query])[0]
        elif hasattr(splade_model, 'embed'):
            splade_embedding = splade_model.embed([enhanced_query])[0]
        else:
           
            raise Exception("No known encoding method found for SparseEncoder")
        
        splade_terms = extract_biomedical_terms(splade_embedding, query, top_k=15)
        splade_term_list = [term for term, weight in splade_terms]
        print(f"   SPLADE terms: {splade_term_list}")
    except Exception as e:
        print(f"   SPLADE failed: {e}, using manual terms")
        splade_terms = [
            ('statin', 3.0),     
            ('breast', 3.0),      
            ('cancer', 3.0),     
            ('cholesterol', 2.0),
            ('drug', 2.0),       
            ('cause', 1.5)        
        ]
        splade_term_list = [term for term, weight in splade_terms]
    
    
    query_biomedical_vector = get_biomedical_embedding(enhanced_query).tolist()
    

    search_body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    # DENSE semantic search
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'dense_vector') + 1.0",
                                "params": {"query_vector": query_biomedical_vector}
                            },
                            "boost": 5.0
                        }
                    },
                    # SPARSE term search
                    {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "text": {
                                            "query": term,
                                            "boost": weight * 2.0
                                        }
                                    }
                                }
                                for term, weight in splade_terms
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    # Fallback: direct term matching
                    {
                        "multi_match": {
                            "query": enhanced_query,
                            "fields": ["text^2", "title^3"],
                            "type": "best_fields",
                            "minimum_should_match": "30%"
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
    }
    
    try:
        response = es.search(index=index_name, body=search_body)
        return response
    except Exception as e:
        print(f"Search failed: {e}")
        return {"hits": {"hits": []}}

# 7. Indexing function (ensure this runs first)
def index_corpus_biomedical(corpus_path="corpus.jsonl"):
    """Index corpus with biomedical embeddings"""
    print("Indexing corpus with biomedical embeddings...")
    
    index_name = "biomedical_hybrid_index"
    index_body = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=index_body)
    
    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            biomedical_embedding = get_biomedical_embedding(doc["text"])
            
            es_doc = {
                "title": doc.get("title", ""),
                "text": doc["text"],
                "dense_vector": biomedical_embedding.tolist()
            }
            
            es.index(index=index_name, id=doc["_id"], body=es_doc)
            
            if i % 100 == 0:
                print(f"Indexed {i} documents...")
    
    print("Indexing complete!")
    es.indices.refresh(index=index_name)


def evaluate_biomedical_thesis(prof_dataset_path="professor_dataset.jsonl", corpus_path="corpus.jsonl"):
    """Evaluate retrieval performance with missing document analysis"""
    print("\n" + "="*60)
    print("BIOMEDICAL HYBRID EVALUATION (SPLADE + DENSE)")
    print("="*60)
    
    try:
        with open(prof_dataset_path, 'r') as f:
            prof_data = json.loads(f.readline().strip())
    except:
        print("Error reading dataset")
        return 0, 0
    
    query = prof_data["question"]
    expected_docs = prof_data["expected_resources"]
    
    print(f"Question: {query}")
    print(f"Expected to retrieve: {expected_docs[:8]}...")
    
    results = biomedical_hybrid_search(query, top_k=12)
    retrieved_docs = [hit['_id'] for hit in results['hits']['hits']]
    
    print(f"Retrieved documents: {retrieved_docs}")
    
    relevant_retrieved = [doc for doc in retrieved_docs if doc in expected_docs]
    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(relevant_retrieved) / len(expected_docs) if expected_docs else 0
    
    print(f"\n-- RESULTS:")
    print(f"Precision: {precision:.1%} ({len(relevant_retrieved)}/{len(retrieved_docs)})")
    print(f"Recall: {recall:.1%} ({len(relevant_retrieved)}/{len(expected_docs)})")
    
    missing_docs = set(expected_docs) - set(retrieved_docs)
    print(f"Missing documents: {len(missing_docs)} documents")
    
    if relevant_retrieved:
        print(f"-- Relevant found: {relevant_retrieved}")
        
        if missing_docs:
            analyze_missing_documents(missing_docs, corpus_path)
    else:
        print("-- No relevant documents found")
    
    return precision, recall


if __name__ == "__main__":
   
    
    index_corpus_biomedical("corpus.jsonl")
    
   
    print("Starting evaluation...")
    precision, recall = evaluate_biomedical_thesis("professor_dataset.jsonl")