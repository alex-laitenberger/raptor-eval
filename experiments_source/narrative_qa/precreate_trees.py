import logging

from raptor import RetrievalAugmentation 
from raptor import RetrievalAugmentationConfig
from raptor.QAModels import OpenAI_QAModel_MultipleChoice
from raptor.EmbeddingModels import SnowflakeArcticEmbeddingModel
from raptor.SummarizationModels import OpenAISummarizationModel

from experiments_source.utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file, openFileWithUnknownEncoding, count_words, remove_html_tags
from datetime import datetime
from config import OPENAI_API_KEY
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import re

from openai import OpenAI

# Hyperparameters
HYPERPARAMS = {
    "tb_max_tokens": 100,  # Max token length for summaries / abstractive chunks in tree
                    #   parameter is also used for chunk length for document chunking (split_text)
}

OPENAI_MODELSTRING = "gpt-4o-mini-2024-07-18"


# Experiment metadata
EXPERIMENT_IDENTIFIER = "raptor-chunk_size_100_Snowflake_NarrativeQA"
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Paths
NARRATIVEQA_PATH = '~/narrativeqa'
STORED_TREES_PATH = f"experiments/artifacts/trees/narrative_qa/test/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
LOG_DIR = "experiments/logs/"
LOG_FILE = f"{LOG_DIR}/preprocess_narrative_{CURRENT_DATE_TIME}.log"


# Ensure necessary directories exist
create_directories([STORED_TREES_PATH, LOG_DIR])

# Load the API key into the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

 # Logging Configuration
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.DEBUG)  # Set the general logging level for the root logger, level is set again for the handlers

# Remove existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")  # Write logs to file
file_handler.setLevel(logging.INFO)  # Set log level for the file handler
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Stream handler (for terminal output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Set log level for the stream handler
stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)




def precreate_tree_for_doc(
                    row_data,
                    RAC):
    document_id = row_data['document_id']
    logging.info(f"Processing document {document_id}...")
    
    # Open and read the file
    file_path = f'{NARRATIVEQA_PATH}/tmp/{document_id}.content'
    expanded_path = os.path.expanduser(file_path)
    content = openFileWithUnknownEncoding(expanded_path)
    if content is not None:
        document_context = remove_html_tags(content).replace('\n', ' ').replace('\t', ' ')
        wordCount = count_words(document_context)
    
        if wordCount > 0:

            # Chunk and embed the document
            rag = RetrievalAugmentation(config=RAC)
            rag.add_documents(document_context)
            logging.info(f"Successfully chunked and embedded document {document_id}.")

            # Save the tree to the artifacts folder
            tree_file_path = f"{STORED_TREES_PATH}/{document_id}"
            rag.save(tree_file_path)
            logging.info(f"Saved trees for document {document_id} at {tree_file_path}.")
            
            # Debug node details
            logging.info(f"Nodes for document_id {document_id}:")
            nodes_list = sorted(rag.tree.all_nodes.values(), key=lambda node: node.index)  # Convert dictionary values to a sorted list

            logging.info("Tree Building Sanity Check:")
            logging.info(f"Total number of nodes: {len(nodes_list)}")
            logging.info(f"Number of Leaf nodes: {len(rag.tree.leaf_nodes.values())}")
            logging.info(f"Number of Root nodes: {len(rag.tree.root_nodes.values())}")
            logging.info(f"Number of Layers: {rag.tree.num_layers}")
            
            nodes_per_layer = {layer: len(nodes) for layer, nodes in rag.tree.layer_to_nodes.items()}
            logging.info(f"nodes_per_layer: {nodes_per_layer }")

            # Log the first 3 nodes
            logging.info("--- First 3 Nodes ---")
            for node in nodes_list[:3]:  # First 3 nodes
                logging.info(f"Index: {node.index}, Text: {node.text}, Embedding: {node.embeddings['EMB'][:5]}...")

            # Log the last 3 nodes
            logging.info("--- Last 3 Nodes ---")
            for node in nodes_list[-3:]:  # Last 3 nodes
                logging.info(f"Index: {node.index}, Text: {node.text}, Embedding: {node.embeddings['EMB'][:5]}...")


def precreate_trees():
    logging.info("Starting precreate_nodes process...")

    # Load data
    
    datasetString = 'test' #test, train, valid
    df = pd.read_csv(f'{NARRATIVEQA_PATH}/documents.csv')
    test_df = df[df['set'] == datasetString]
    #print(test_df.head(2))

    qaps_df = pd.read_csv(f'{NARRATIVEQA_PATH}/qaps.csv')

    #print(qaps_df.head(1))

    # Initialize models
    logging.info("Initializing models...")
    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    qa_model = OpenAI_QAModel_MultipleChoice(modelString=OPENAI_MODELSTRING, client=openAI_client)
    embedding_model = SnowflakeArcticEmbeddingModel()
    summarization_model = OpenAISummarizationModel(modelString=OPENAI_MODELSTRING, client=openAI_client)

    
    # Initialize RAG pipeline
    RAC = RetrievalAugmentationConfig(summarization_model=summarization_model, qa_model=qa_model, embedding_model=embedding_model, tb_max_tokens=HYPERPARAMS["tb_max_tokens"])

    created_trees_counter = 0

    failed_docs = []

    for index, row_data in test_df.iterrows():
        try:
            precreate_tree_for_doc(row_data, RAC)
        except Exception as e:
            document_id = row_data['document_id']
            logging.exception(f"[ERROR] Failed to create tree for document {document_id}: {e}")
            failed_docs.append(document_id)

    logging.info("Finished Tree building.")

    if failed_docs:
        logging.warning(f"Tree creation failed for {len(failed_docs)} documents: {failed_docs}")


if __name__ == "__main__":
    precreate_trees()
