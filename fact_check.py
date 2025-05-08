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
# rm = SentenceTransformersRM(model="all-MiniLM-L6-v2",
#                             device='cuda')
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")


# Configure vector store with specific index paths
# index_base_dir = "./benchmark/index"
# index_path = os.path.join(index_base_dir, "index", "index")
# vecs_path = os.path.join(index_base_dir, "index", "vecs")

# Configure vector store with specific index paths
index_base_dir = "/mnt/homes/ktle/lotus-ai/benchmark/datasets/fever/index"
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
dataset = "fever"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)
# data_path = util.unzip("/mnt/homes/ktle/lotus-ai/benchmark/datasets/fever.zip", out_dir)

def data_load(sample=10):
    corpus, queries, qrels = GenericDataLoader(data_folder="/mnt/homes/ktle/lotus-ai/benchmark/datasets/fever/").load(split="dev")
    logger.info(f"Dataset loaded. Corpus size: {len(corpus)}, Queries size: {len(queries)}")
    # === Convert corpus to DataFrame ===
    corpus_df = pd.DataFrame([{
        "doc_id": doc_id,
        "text": doc['text']
    } for doc_id, doc in corpus.items()])
    corpus_df = corpus_df.sample(n=1000, random_state=42).reset_index(drop=True)
    corpus_df.attrs["index_dirs"] = {"text": index_base_dir}

    # corpus_df = corpus_df.sem_index('text', "/mnt/homes/ktle/lotus-ai/benchmark/datasets/fever/index")
    # === Convert queries to DataFrame with gold label
    queries_df = pd.DataFrame([{
        "query_id": query_id,
        "claim": query,
    } for query_id, query in queries.items()])

    # === Sample n_sample queries
    queries_df = queries_df.sample(n=sample, random_state=42).reset_index(drop=True)

    return corpus_df, queries_df

# def fact_check(corpus_df, queries_df):
#     queries_df = queries_df.sem_map(
#         user_instruction="Rewrite the claim into a search query to retrieve relevant Wikipedia evidence. Claim: {claim}",
#         suffix="claim_map"
#     )

#     results = []
#     for _, row in queries_df.iterrows():
#         query_text = row["claim_map"]
#         qid = row["query_id"]

#         # Search the corpus for each mapped query
#         search_results = corpus_df.sem_search(
#             col_name="text",
#             query=query_text,
#             K=5
#         ).copy()

#         # Tag with query_id and original claim
#         search_results["query_id"] = qid
#         search_results["claim"] = row["claim"]
#         search_results["mapped_query"] = query_text
#         results.append(search_results)

#     retrieved_df = pd.concat(results).reset_index(drop=True)
#     # === FILTER step ===
#     retrieved_df = retrieved_df.sem_filter(
#         user_instruction="Does the following document support the claim '{claim}'?\n\n{text}",
#         suffix="_filter"
#     )

#     print(retrieved_df)
#     return retrieved_df


def fact_check(corpus_df, queries_df):
    """
    Perform fact-checking using the Lotus pipeline.
    
    Args:
        corpus_df (pd.DataFrame): Corpus dataframe with documents
        queries_df (pd.DataFrame): Queries dataframe with claims to check
        
    Returns:
        pd.DataFrame: Results dataframe with fact-checking information
    """
    logger.info(f"Starting fact-checking process for {len(queries_df)} claims...")
    
    try:
        # Step 1: Query rewriting
        logger.info("Performing query rewriting...")
        queries_df = queries_df.sem_map("Rewrite the claim into a semantic search query question to retrieve relevant Wikipedia evidence. Claim: {claim}. Only respond with the query and do not need include site:wikipedia.org.")
        logger.info("Query rewriting completed successfully")
    except Exception as e:
        logger.error(f"Error during query rewriting: {e}")
        logger.warning("Falling back to original claim text for retrieval")    
    results = []
    # Step 2: Document retrieval for each claim
    for idx, row in queries_df.iterrows():
        query_id = row["query_id"]
        claim = row["claim"]
        query_text = row["_map"]
        
        logger.info(f"Processing claim {idx+1}/{len(queries_df)}: ID={query_id}")
        logger.info(f"Original claim: {claim}")
        logger.info(f"Mapped query: {query_text}")
        
        # Search the corpus for documents relevant to the query
        search_results = corpus_df.sem_search(
            'text',
            f"Which text answer this question: {query_text}?",
            K=5,
        )
        logger.info(f"Retrieved {len(search_results)} documents")
        
        logger.info(f'Started filtering process for claim: {claim}')
        supporting_search_results = search_results.sem_filter(
            f"Text provide evidence for {claim}?"
        )
                # If you have a unique identifier column like 'doc_id'
        supporting_ids = supporting_search_results['doc_id'].tolist()
        unsupporting_search_results = search_results[~search_results['doc_id'].isin(supporting_ids)]
        results.append({
            "claim": claim,
            "mapped_query": query_text,
            "supporting_texts": "\n---\n".join(supporting_search_results['text'].tolist()),
            "non_supporting_texts": "\n---\n".join(unsupporting_search_results['text'].tolist()),
        })
        
    return pd.DataFrame(results)


# === Run Everything ===
if __name__ == "__main__":
    corpus_df, queries_df = data_load(5)
    results_df = fact_check(corpus_df, queries_df)
    results_df.to_csv("fever_fact_check_output.csv", index=False)