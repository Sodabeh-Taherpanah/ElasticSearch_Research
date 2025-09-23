# ElasticSearch_Research

This repository explores whether **Elasticsearch retrieval improvements can positively affect Retrieval-Augmented Generation (RAG) quality**. It investigates the impact of different Elasticsearch retrieval strategies — **sparse , dense (neural embeddings), hybrid, and pure neural search** — on the quality of large language model (LLM) outputs.

The project compares:
- **Traditional sparse retrieval (BM25)**  
- **Dense retrieval with embeddings (E5, SPLADE)**  
- **Hybrid retrieval (sparse + dense)**  
- **Pure neural retrieval pipelines**  

Finally, the results are analyzed in the context of **RAG pipelines**, showing how retrieval choice influences the **quality, accuracy, and relevance** of generated answers.



<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4a963302-dcec-4a03-a9e8-42c22257b6e0" />

## Project Structure

dataSet/: Contains datasets used for indexing and evaluation.

neural/: Scripts for indexing documents using neural embeddings and evaluating retrieval performance.

simpleRAG/: Implements a simple Retrieval-Augmented Generation pipeline.

head_db.py: Utility script for managing Elasticsearch indices and document metadata.

neural_retrieval_results.json: Stores results from neural retrieval evaluations.

## Installation

Ensure you have the following dependencies installed:

pip install elasticsearch sentence-transformers torch tqdm matplotlib

##Neural Indexing

The neural/ directory contains scripts for indexing biomedical documents using dense vector embeddings from the intfloat/e5-large-v2 model. This approach enables semantic search capabilities in Elasticsearch.

To index documents:
python neural_indexing.py
This will create an Elasticsearch index optimized for neural retrieval.

## Retrieval-Augmented Generation (RAG)

The simpleRAG/ directory demonstrates a simple RAG pipeline, integrating Elasticsearch for document retrieval and a generative model for answer generation.

To run the RAG pipeline:

python rag_pipeline.py


## Evaluation

The neural/ directory also includes scripts for evaluating retrieval performance using metrics such as Precision, Recall, and F1-Score. Evaluation is performed on a sample dataset, and results are saved in neural_retrieval_results.json.

To evaluate retrieval performance:

python evaluate_retrieval.py


## Visualizations

Matplotlib is used for visualizing retrieval performance metrics. You can generate plots by running:


python visualize_results.py



