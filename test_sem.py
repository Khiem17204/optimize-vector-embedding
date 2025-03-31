# lotus model
import pandas as pd
import lotus
from lotus.models import LM
import logging
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configure lotus LLM

# env variable
os.environ['DEEPSEEK_API_KEY'] = "sk-520ee20b73934d8ca49d3d16318d9d40"
lm = LM(model="deepseek/deepseek-chat")
logger.info("Initializing Lotus LLM...")
# lm = LM(model="ollama/llama3.2:3b")
lotus.settings.configure(lm=lm)
logger.info("Lotus LLM initialized")

# Create small test data
test_docs = [
    {"id": "1", "content": "Title: Test Document 1\n\nAbstract: This is a test document about dog."},
    {"id": "2", "content": "Title: Test Document 2\n\nAbstract: This document discusses neural networks."}
]
test_query = "Find papers about machine learning"

# Convert to DataFrames
logger.info("Creating test DataFrames...")
docs_df = pd.DataFrame(test_docs)
logger.info(f"Test corpus:\n{docs_df}")

# Try semantic search
logger.info(f"\nTrying semantic search with query: {test_query}")
try:
    logger.info("Starting sem_topk...")
    results = docs_df[['content']].sem_topk(test_query, K=1)
    logger.info("\nSearch results:")
    logger.info(results)
except Exception as e:
    logger.error(f"Error during semantic search: {str(e)}")

# Print LLM usage
logger.info("\nPrinting LLM usage statistics...")
lm.print_total_usage() 