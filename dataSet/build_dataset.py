#.  plan
#  data Preparation: Process the downloaded SciDocs files (corpus.jsonl, queries.jsonl, qrels/test.tsv) into your professor's required format.

# Modern Retrieval Setup: Index this data using a state-of-the-art method (SPLADE) in Elasticsearch.

# Evaluation: Use the prepared dataset to evaluate the retrieval performance.

# so. first 

#     Build Dataset: python 1_build_professor_dataset.py

#         Creates professor_dataset.jsonl for your thesis.

#     Modern Indexing: python 2_splade_indexing.py

#         Indexes the corpus using a 2024 SPLADE model.

#     Testing: python 3_modern_retrieval_test.py

#         Tests the modern system with your professor's format.


#This script reads your downloaded files and creates a new JSON file #(professor_dataset.jsonl) where each line is a perfect example


import pandas as pd
import json

# 1. Load the downloaded files
print("Loading downloaded SciDocs files...")
corpus_df = pd.read_json('corpus.jsonl', lines=True)
queries_df = pd.read_json('queries.jsonl', lines=True)

# Load the QRELS file (the "answer key")
# This file tells us which document is the correct answer for each query
qrels_df = pd.read_csv('test.tsv', sep='\t', header=None, names=['query_id', 'corpus_id', 'score'])
print("QRELS columns:", qrels_df.columns.tolist())


professor_dataset = []

# 3. Convert all IDs to strings to ensure consistent matching
corpus_df['_id'] = corpus_df['_id'].astype(str)
queries_df['_id'] = queries_df['_id'].astype(str)
qrels_df['query_id'] = qrels_df['query_id'].astype(str)
qrels_df['corpus_id'] = qrels_df['corpus_id'].astype(str)

# 4. Get the UNIQUE query IDs from the QRELS file
# This prevents the same question from being added multiple times
unique_query_ids = qrels_df['query_id'].unique()
print(f"\nFound {len(unique_query_ids)} unique queries in the QRELS file.")

for query_id in unique_query_ids:
    # A. Find the query text for this ID
    query_row = queries_df[queries_df['_id'] == query_id]
    if query_row.empty:
        print(f"Warning: Query ID {query_id} not found in queries.jsonl. Skipping.")
        continue
        
    question_text = query_row['text'].iloc[0]

    # B. Find ALL relevant document IDs for this query from the QRELS
    # This gets all correct answers for this question
    relevant_docs = qrels_df[qrels_df['query_id'] == query_id]
    expected_resources = relevant_docs['corpus_id'].tolist()
    
    # C. Create ONE dataset entry for this question
    # The "answer" will be the text of the FIRST relevant document
    # (or you could concatenate multiple if needed)
    primary_doc_id = expected_resources[0]
    primary_doc_row = corpus_df[corpus_df['_id'] == primary_doc_id]
    
    if primary_doc_row.empty:
        print(f"Warning: Document ID {primary_doc_id} not found in corpus. Skipping query {query_id}.")
        continue
        
    answer_text = primary_doc_row['text'].iloc[0]

   
    custom_row = {
        "question": question_text,
        "answer": answer_text[:1000] + "..." if len(answer_text) > 1000 else answer_text,  # Optional: shorten very long answers
        "expected_resources": expected_resources,  # List of all correct document IDs
        "expected_resource_pages": [1] * len(expected_resources)  # Creates [1, 1, 1] for 3 resources
    }
    
    professor_dataset.append(custom_row)

# 5. Save the custom dataset to a JSONL file
output_file = 'professor_dataset.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for row in professor_dataset:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f"\nDone! Created professor's dataset with {len(professor_dataset)} question-answer pairs.")
print(f"Saved to: {output_file}")

# Show a sample of what was created
if professor_dataset:
    print("\nExample of the first entry:")
    sample_entry = professor_dataset[0].copy()
    # Shorten the answer in the sample display for readability
    if len(sample_entry['answer']) > 200:
        sample_entry['answer'] = sample_entry['answer'][:200] + "..."
    print(json.dumps(sample_entry, indent=2, ensure_ascii=False))
else:
    print("\nERROR: The dataset is empty. Check the warnings above.")