# # BEIR data loading and evaluation
# from beir import util, LoggingHandler
# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.evaluation import EvaluateRetrieval

# # lotus model
# import pandas as pd
# import lotus
# from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
# from lotus.vector_store import FaissVS
# import numpy as np

# import logging
# import pathlib, os
# import time

# #### Just some code to print debug information to stdout
# logging.basicConfig(format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO,
#                     handlers=[LoggingHandler()])
# logger = logging.getLogger(__name__)

# lm = LM(model="deepseek/deepseek-chat")
# # lm = LM(model='ollama/llama3.2:3b')
# # rm = SentenceTransformersRM(model="all-MiniLM-L6-v2",
# #                             device='cuda')
# rm = SentenceTransformersRM(model="/home/ktle_umass_edu/optimize-vector-embedding/e5-base-v2-local")

# reranker = CrossEncoderReranker(model="/home/ktle_umass_edu/optimize-vector-embedding/mxbai-rerank-local")


# # Configure vector store with specific index paths
# # index_base_dir = "./benchmark/index"
# # index_path = os.path.join(index_base_dir, "index", "index")
# # vecs_path = os.path.join(index_base_dir, "index", "vecs")

# # Configure vector store with specific index paths
# vs = FaissVS()
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# # Configure all components
# lotus.settings.configure(lm=lm, rm=rm, reranker=reranker, vs=vs)
# print("Lotus LLM initialized with all components")

# corpus, queries, qrels = GenericDataLoader(data_folder=f"{out_dir}/fever/").load(split="dev")
# logger.info(f"Dataset loaded. Corpus size: {len(corpus)}, Queries size: {len(queries)}")

# corpus_df = pd.DataFrame([{
#     "text": doc['text']
# } for _, doc in corpus.items()])
# corpus_df = corpus_df.sem_index('text', f"{out_dir}/fever/index/")



import argparse
import logging
import pathlib
import os
import pandas as pd
import numpy as np
import lotus
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
from lotus.vector_store import FaissVS

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--split-id", type=int, required=True, help="Index of this data split (0-based)")
parser.add_argument("--num-splits", type=int, default=10, help="Total number of splits")
args = parser.parse_args()

# --- Logging setup ---
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# --- Load models ---
lm = LM(model="deepseek/deepseek-chat")
rm = SentenceTransformersRM(model="/home/ktle_umass_edu/optimize-vector-embedding/e5-base-v2-local")
reranker = CrossEncoderReranker(model="/home/ktle_umass_edu/optimize-vector-embedding/mxbai-rerank-local")

vs = FaissVS()
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
lotus.settings.configure(lm=lm, rm=rm, reranker=reranker, vs=vs)
logger.info("Lotus LLM initialized with all components")

# --- Load dataset ---
corpus, queries, qrels = GenericDataLoader(data_folder=f"{out_dir}/fever/").load(split="dev")
logger.info(f"Dataset loaded. Corpus size: {len(corpus)}, Queries size: {len(queries)}")

# --- Split corpus ---
all_texts = [doc["text"] for _, doc in corpus.items()]
total = len(all_texts)
split_size = total // args.num_splits
start = args.split_id * split_size
end = total if args.split_id == args.num_splits - 1 else (args.split_id + 1) * split_size

chunk = all_texts[start:end]
logger.info(f"Split {args.split_id}: Indexing {len(chunk)} docs (range {start} to {end})")

# --- Create DataFrame and Index ---
df_chunk = pd.DataFrame({"text": chunk})
index_dir = os.path.join(out_dir, f"fever/index_part_{args.split_id}")
df_chunk.sem_index("text", index_dir)
logger.info(f"Index for split {args.split_id} written to: {index_dir}")
