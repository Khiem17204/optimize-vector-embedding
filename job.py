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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

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
vs = FaissVS()
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# Configure all components
lotus.settings.configure(lm=lm, rm=rm, reranker=reranker, vs=vs)
print("Lotus LLM initialized with all components")

corpus, queries, qrels = GenericDataLoader(data_folder=f"{out_dir}/fever/").load(split="dev")
logger.info(f"Dataset loaded. Corpus size: {len(corpus)}, Queries size: {len(queries)}")

corpus_df = pd.DataFrame([{
    "text": doc['text']
} for _, doc in corpus.items()])
corpus_df = corpus_df.sem_index('text', f"{out_dir}/fever/index/")
