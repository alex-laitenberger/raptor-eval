import logging

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

OPENAI_MODELSTRING = "gpt-4o-mini-2024-07-18"

# Hyperparameters
HYPERPARAMS = {
    "tb_max_tokens": 100,  # Max token length for summaries / abstractive chunks in tree
                    #   parameter is also used for chunk length for document chunking (split_text)
}

# Experiment metadata
EXPERIMENT_IDENTIFIER = "raptor-chunk_size_100_Snowflake_Quality_dev"
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Paths
STORED_TREES_PATH = f"experiments/artifacts/trees/quality/dev/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
STORED_ANSWERS_PATH = "experiments/artifacts/answers/quality/"
LOG_DIR = "experiments/logs/"
LOG_FILE = f"{LOG_DIR}/preprocess_quality_{CURRENT_DATE_TIME}.log"


# Ensure necessary directories exist
create_directories([STORED_TREES_PATH, STORED_ANSWERS_PATH, LOG_DIR])

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
    preprocessed_path = "data/quality/preprocessed/QuALITY.v1.0.1.htmlstripped_dev_preprocessed.json"
    data_dict = load_json_file(preprocessed_path)

    # Initialize models
    logging.info("Initializing models...")
    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    qa_model = OpenAI_QAModel_MultipleChoice(modelString=OPENAI_MODELSTRING, client=openAI_client)
    embedding_model = SnowflakeArcticEmbeddingModel()
    summarization_model = OpenAISummarizationModel(modelString=OPENAI_MODELSTRING, client=openAI_client)

    # Initialize RAG pipeline
    RAC = RetrievalAugmentationConfig(summarization_model=summarization_model, qa_model=qa_model, embedding_model=embedding_model, tb_max_tokens=HYPERPARAMS["tb_max_tokens"])

    created_trees_counter = 0

    for article_id, content in data_dict.items():
        try:
            logging.debug(f"Processing document {article_id}...")

            # Extract document context
            document_context = content['article']

            # Chunk and embed the document
            rag = RetrievalAugmentation(config=RAC)
            rag.add_documents(document_context)
            logging.info(f"Successfully chunked and embedded document {article_id}.")

            # Save the tree to the artifacts folder
            tree_file_path = f"{STORED_TREES_PATH}/{article_id}"
            rag.save(tree_file_path)
            logging.info(f"Saved trees for document {article_id} at {tree_file_path}.")
            
            # Debug node details
            logging.info(f"Nodes for article_id {article_id}:")
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
            logging.exception(f"Error processing article_id {article_id}: {e}")

if __name__ == "__main__":
    precreate_trees()
