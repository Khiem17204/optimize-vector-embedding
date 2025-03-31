# BEIR data loading and evaluation
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# lotus model
import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
from lotus.vector_store import FaissVS
import numpy as np

import logging
import pathlib, os
import time

# configure lotus LLM
print("Initializing Lotus LLM...")
os.environ['DEEPSEEK_API_KEY'] = ""

lm = LM(model="deepseek/deepseek-chat")
# lm = LM(model='ollama/llama3.2:3b')
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")
# Configure vector store with specific index paths
index_base_dir = "./benchmark/index"
index_path = os.path.join(index_base_dir, "index", "index")
vecs_path = os.path.join(index_base_dir, "index", "vecs")

# Configure vector store with specific index paths
index_base_dir = "/mnt/homes/ktle/lotus-ai/benchmark/index"
vs = FaissVS()
vs.load_index(index_base_dir)
# Configure all components
lotus.settings.configure(lm=lm, rm=rm, reranker=reranker, vs=vs)
print("Lotus LLM initialized with all components")

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
logger.info(f"Dataset loaded. Corpus size: {len(corpus)}, Queries size: {len(queries)}")

# Convert corpus to DataFrame format for Lotus
logger.info("Converting corpus to DataFrame...")
corpus_df = pd.DataFrame([{
    'id': doc_id,
    'content': f"Title: {doc['title']}\n\nAbstract: {doc['text']}"  
} for doc_id, doc in corpus.items()])
logger.info(f"Corpus DataFrame created with {len(corpus_df)} rows")

# Convert queries to DataFrame format
queries_df = pd.DataFrame([{
    'id': query_id,
    'query': f"Which papers relate the most to {query}?"
} for query_id, query in queries.items()])
# logger.info("Using only the first query for testing")

# Function to perform retrieval using Lotus sem_k
def lotus_sem_topk(queries_df, corpus_df, k=10):
    results = {}
    total_queries = len(queries_df)
    
    for idx, row in queries_df.iterrows():
        query_id = row['id']
        query_text = row['query']
        logger.info(f"\nProcessing query {idx + 1}/{total_queries}")
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Query text: {query_text}")
        
        try:
            # Use sem_k to find similar documents
            logger.info("Starting semantic search...")
            start_time = time.time()
            similar_docs = corpus_df[['content']].sem_topk(query_text, K=k)
            end_time = time.time()
            logger.info(f"Search completed in {end_time - start_time:.2f} seconds")
            
            # Format results as required by BEIR evaluation
            scores = {}
            for idx, score in zip(similar_docs.index, similar_docs['score']):
                doc_id = corpus_df.loc[idx, 'id']
                scores[doc_id] = float(score)
                # Print the matched document for inspection
                doc_content = corpus_df.loc[idx, 'content']
                logger.info(f"Matched document {doc_id} (score: {score:.3f}):")
                logger.info(f"{doc_content[:200]}...")
            
            results[query_id] = scores
            logger.info(f"Query {query_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            raise
    
    return results

# Function to perform retrieval using sem_index and sem_search with reranking
def lotus_search_and_rank(queries_df, corpus_df, k=10, n_rerank=5):
    results = {}
    total_queries = len(queries_df)
    
    # Create semantic index
    # logger.info("Creating semantic index...")
    # corpus_df = corpus_df.sem_index('content', index_dir)
    corpus_df.attrs["index_dirs"] = {"content": index_base_dir}
    for idx, row in queries_df.iterrows():
        query_id = str(row['id'])  # Ensure query_id is a string
        query_text = row['query']
        logger.info(f"\nProcessing query {idx + 1}/{total_queries}")
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Query text: {query_text}")
        
        try:
            # Use sem_search with reranking
            logger.info("Starting semantic search with reranking...")
            start_time = time.time()
            search_results = corpus_df.sem_search(
                'content',
                query_text,
                K=k,
                n_rerank=n_rerank,
                return_scores=True
            )
            end_time = time.time()
            logger.info(f"Search and rerank completed in {end_time - start_time:.2f} seconds")
            
            # Format results as required by BEIR evaluation
            scores = {}
            for result_idx, row in search_results.iterrows():
                # Ensure doc_id is converted to string
                doc_id = str(corpus_df.loc[result_idx, 'id'])
                score = float(row['vec_scores_sim_score'])
                scores[doc_id] = score
                
                # Optional: Print matched documents
                doc_content = corpus_df.loc[result_idx, 'content']
                logger.info(f"Matched document {doc_id} (score: {score:.3f}):")
                logger.info(f"{doc_content[:200]}...")
            
            results[query_id] = scores
            logger.info(f"Query {query_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            raise
    
    return results

# Perform retrieval using Lotus
logger.info("Starting retrieval process with search and rank...")
try:
    # Use the new search and rank function
    # results = lotus_search_and_rank(queries_df, corpus_df, k=100, n_rerank=20)
    result = lotus_search_and_rank(queries_df, corpus_df)
    logger.info("Retrieval completed successfully")
except Exception as e:
    logger.error(f"Retrieval failed: {str(e)}")
    raise

#### Evaluate using BEIR's evaluation metrics
logger.info("Starting evaluation...")
evaluator = EvaluateRetrieval()
ndcg, _map, recall, precision = evaluator.evaluate(qrels, result, [1, 3, 10])
mrr = evaluator.evaluate_custom(qrels, result, [10], metric="mrr")
logger.info("Evaluation completed")

### Save results
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), result)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)

# Print total LM usage
logger.info("Printing LLM usage statistics...")
lm.print_total_usage()