import logging
import psutil
from raptor import RetrievalAugmentation 
from raptor import RetrievalAugmentationConfig
from raptor.QAModels import OpenAI_QAModel_MultipleChoice
from raptor.EmbeddingModels import SnowflakeArcticEmbeddingModel
from raptor.SummarizationModels import OpenAISummarizationModel
from experiments_source.utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file
from datetime import datetime
from config import OPENAI_API_KEY
import os

from openai import OpenAI

# Experiment metadata
EXPERIMENT_IDENTIFIER = "raptor-rag-chunk_size_100-Emb_Snowflake-Sum_GPT4o-mini"
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Paths
PREPROCESSED_PATH = "data/infinity_bench/preprocessed/longbook_choice_eng_preprocessed.json"
STORED_TREES_PATH = f"experiments/artifacts/trees/infinity_bench/longbook_choice_eng/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
STORED_ANSWERS_PATH = "experiments/artifacts/answers/infinity_bench/longbook_choice_eng/"
LOG_DIR = "experiments/logs/"
LOG_FILE = f"{LOG_DIR}/infinity_bench_longbook_choice_eng_precreate_trees_{CURRENT_DATE_TIME}.log"

OPENAI_QA_MODELSTRING = "gpt-4o-mini-2024-07-18" #openAI model used generally for QA (not actually used in precreate trees step)
OPENAI_SUM_MODELSTRING = "gpt-4o-mini-2024-07-18" #openAI model used to create cluster summaries in precreate_trees step

# Ensure necessary directories exist
create_directories([STORED_TREES_PATH, STORED_ANSWERS_PATH, LOG_DIR])

# Load the API key into the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Logging Configuration
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the logging level for the logger

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

# Hyperparameters
HYPERPARAMS = {
    "tb_max_tokens": 100,  # Max token length for summaries / abstractive chunks in tree
                    #   parameter is also used for chunk length for document chunking (split_text)
}

def log_open_file_descriptors():
    pid = os.getpid()
    process = psutil.Process(pid)
    open_files = process.open_files()
    logging.info(f"Open file descriptors: {len(open_files)}")

def get_file_list(folder_path):
    """Get a list of files in a folder, excluding non-files and hidden files."""
    return [
        file
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file)) and not file.startswith(".")
    ]

def precreate_trees():
    logging.info("Starting precreate_trees process...")

    # Load preprocessed dataset
    grouped_data = load_json_file(PREPROCESSED_PATH)

    # Initialize models
    logging.info("Initializing models...")
    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    qa_model = OpenAI_QAModel_MultipleChoice(modelString=OPENAI_QA_MODELSTRING, client=openAI_client)
    embedding_model = SnowflakeArcticEmbeddingModel()
    summarization_model = OpenAISummarizationModel(modelString=OPENAI_SUM_MODELSTRING, client=openAI_client)

    # Initialize RAG Config
    RAC = RetrievalAugmentationConfig(summarization_model=summarization_model, qa_model=qa_model, embedding_model=embedding_model, tb_max_tokens=HYPERPARAMS["tb_max_tokens"])

    created_trees_counter = 0

    for doc_id, doc_data in grouped_data.items():

        try:
            logging.info(f"Processing document {doc_id}...")

            log_open_file_descriptors()

            # Extract document context
            document_context = doc_data["context"]

            # Chunk and embed the document
            rag = RetrievalAugmentation(config=RAC)
            rag.add_documents(document_context)
            logging.info(f"Successfully chunked and embedded document {doc_id}.")

            # Save the tree to the artifacts folder
            tree_file_path = f"{STORED_TREES_PATH}/{doc_id}"
            rag.save(tree_file_path)
            logging.info(f"Saved trees for document {doc_id} at {tree_file_path}.")
            
            # Debug node details
            logging.info(f"Nodes for doc_id {doc_id}:")
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

            created_trees_counter += 1

        except Exception as e:
            # Log errors for this document
            logging.error(f"doc_id: {doc_id}, error: {str(e)}")
            print(f"Error processing doc_id {doc_id}: {e}")


    logging.info(f"Completed precreate_trees, created {created_trees_counter} trees.")


if __name__ == "__main__":
    precreate_trees()
