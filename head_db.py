import pandas as pd
import json

# -----------------------------
# 1. Load files
# -----------------------------
corpus_df = pd.read_json('corpus.jsonl', lines=True)
queries_df = pd.read_json('queries.jsonl', lines=True)
qrels_df = pd.read_csv('test.tsv', sep='\t', header=0)  # header row included in file

# -----------------------------
# 2. Ensure IDs are strings
# -----------------------------
corpus_df['_id'] = corpus_df['_id'].astype(str)
queries_df['_id'] = queries_df['_id'].astype(str)
qrels_df['query-id'] = qrels_df['query-id'].astype(str)
qrels_df['corpus-id'] = qrels_df['corpus-id'].astype(str)

# -----------------------------
# 3. Print head of each file
# -----------------------------
print("=== Corpus (first 5 rows) ===")
print(corpus_df.head(), "\n")

print("=== Queries (first 5 rows) ===")
print(queries_df.head(), "\n")

print("=== QRELS/Test file (first 5 rows) ===")
print(qrels_df.head(), "\n")

# -----------------------------
# 4. Build sample professor dataset
# -----------------------------
professor_dataset = []

for sample_query_id in qrels_df['query-id'].unique():
    query_row = queries_df[queries_df['_id'] == sample_query_id]
    if query_row.empty:
        print(f"⚠️ Warning: Query ID {sample_query_id} not found in queries.jsonl")
        continue
    question = query_row['text'].iloc[0]

    doc_ids = qrels_df[qrels_df['query-id'] == sample_query_id]['corpus-id'].tolist()
    answer_doc_id = doc_ids[0]
    answer_row = corpus_df[corpus_df['_id'] == answer_doc_id]
    if answer_row.empty:
        print(f"⚠️ Warning: Document ID {answer_doc_id} not found in corpus.jsonl")
        continue
    answer = answer_row['text'].iloc[0]

    professor_dataset.append({
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "expected_resources": doc_ids,
        "urls": [corpus_df[corpus_df['_id']==doc_id]['metadata'].iloc[0].get('url', None) 
                 for doc_id in doc_ids]
    })

# -----------------------------
# 5. Print sample professor dataset
# -----------------------------
print(f"✅ Created sample professor dataset with {len(professor_dataset)} entries")
print(json.dumps(professor_dataset[:2], indent=2))  # show first 2 examples
