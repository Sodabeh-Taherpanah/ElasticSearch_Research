This repository demonstrates the application of Elasticsearch in biomedical information retrieval, focusing on pure neural search and Retrieval-Augmented Generation (RAG) techniques. It includes indexing biomedical documents using dense vector embeddings and evaluating retrieval performance.


** Project Structure

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



