import os
import re
import time
import hashlib
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

# Data Handling
import numpy as np
import pandas as pd
from tqdm import tqdm

# Environment Variables
from dotenv import load_dotenv

# Caching
from functools import lru_cache

# Text processing
import nltk
from nltk.tokenize import sent_tokenize
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not found. Falling back to HuggingFace tokenizer for token counting.")

# Vector database
import chromadb
# Use SentenceTransformer embedding function for LOCAL embeddings (not API)
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Models and embeddings
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline

# Hugging Face Hub Client (Optional)
try:
    from huggingface_hub import InferenceClient, HfApi
    # Note: TextGenerationOutput is not imported as it's not needed and may not exist in all versions
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError # Added GatedRepoError
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError as e:
    HUGGINGFACE_HUB_AVAILABLE = False
    InferenceClient = None # Define as None if not available
    HfApi = None
    RepositoryNotFoundError = None
    GatedRepoError = None # Define error type as None if not available
    logging.warning(f"huggingface_hub library not found: {e}. Inference Client functionality will be disabled.")

# Google Gemini API Client (Optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    GEMINI_AVAILABLE = False
    genai = None
    logging.warning(f"google-generativeai library not found: {e}. Gemini API functionality will be disabled.")


# Text ranking and search
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# For evaluation
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge library not found. Generation evaluation metrics will be limited.")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.warning("datasets library not found. Default evaluation dataset cannot be loaded.")


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Try to download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt' tokenizer model. Sentence tokenization might fail. Error: {e}")

# --- Dataclasses ---

@dataclass
class Document:
    """Class representing a document or chunk of a document"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None # Storing embedding is optional here as Chroma handles it

    def __post_init__(self):
        if not self.id:
            # Generate a default ID if not provided, based on text hash for potential deduplication
            self.id = f"doc_{hashlib.md5(self.text.encode()).hexdigest()}"

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    # Model settings - UPGRADED FOR SOTA
    embedding_model_name: str = "intfloat/e5-large-v2"  # Dense retriever upgrade
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model_name: str = "google/flan-t5-base" # Default for local loading OR used as fallback ID

    # --- API Provider Options ---
    use_inference_client: bool = False # Set True to use HF Inference API
    use_gemini_api: bool = False  # Set True to use Google Gemini API
    use_openrouter_api: bool = False  # Set True to use OpenRouter API (recommended!)
    
    # Specific model for inference client (can be different from llm_model_name)
    inference_client_model: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
    gemini_model_name: str = "gemini-2.5-flash"  # Latest and best free Gemini model
    openrouter_model_name: str = "meta-llama/llama-3.2-3b-instruct:free"  # Best for RAG!
    
    # API Keys
    hf_api_token_env_var: str = "HUGGINGFACE_API_KEY" # Environment variable name for the token
    gemini_api_key_env_var: str = "GEMINI_API_KEY"  # Environment variable for Gemini
    openrouter_api_key_env_var: str = "OPENROUTER_API_KEY"  # Environment variable for OpenRouter

    # Chunking settings - OPTIMIZED FOR HOTPOTQA
    chunk_strategy: str = "paragraph_sentence" # Options: "fixed", "paragraph_sentence", "parent_child"
    chunk_size: int = 256  # BALANCED for precision + context (was 128, too small; was 512, too large)
    chunk_overlap: int = 64  # 25% overlap (was 64)
    use_parent_child_chunking: bool = False  # Enable parent-child hierarchical chunking (better precision + context)

    # Retrieval settings - OPTIMIZED FOR SOTA
    top_k: int = 10 # Retrieve more initially for reranker
    rerank_top_n: int = 3 # Number of documents after reranking (SOTA standard)
    similarity_threshold: float = 0.0 # Minimum similarity score (less useful with reranking)
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.7 # Weight for semantic search in hybrid fusion (0=keyword, 1=semantic)
    use_reranking: bool = True  # MUST BE TRUE FOR SOTA
    
    # Faithfulness verification
    use_grounding_check: bool = True  # Enable context overlap verification
    min_context_overlap: float = 0.0  # EMERGENCY: Disabled grounding check (was killing valid answers!)

    # Database settings
    collection_name: str = "rag_documents_default" # Default collection name
    persist_directory: str = "./chroma_db_default" # Default persistence path

    # System settings
    cache_size: int = 1024 # LRU cache size for retrieved documents
    use_gpu: bool = torch.cuda.is_available() # For local models/reranker
    llm_max_new_tokens: int = 75  # CONCISE BUT NOT TOO RESTRICTIVE (was 50, too short; was 256, too long)
    llm_temperature: float = 0.0  # DETERMINISTIC for factual answers (was 0.3)
    llm_top_p: float = 0.9 # Generation top-p sampling

# --- Core Components ---

class TextProcessor:
    """Class for text processing operations like chunking and token counting."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._tokenizer = None
        self._llm_tokenizer = None # Use LLM tokenizer for fallback counting

    # (Tokenizer properties - unchanged from previous version)
    @property
    def tokenizer(self):
        """Lazy load the tokenizer for token counting."""
        if self._tokenizer is None:
            if TIKTOKEN_AVAILABLE:
                try:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.info("Using tiktoken cl100k_base for token counting.")
                except Exception as e:
                    logger.warning(f"Failed to load tiktoken cl100k_base: {e}. Falling back to LLM tokenizer.")
                    self._tokenizer = self.llm_tokenizer # Fallback
            else:
                 self._tokenizer = self.llm_tokenizer # Fallback if tiktoken not installed
        return self._tokenizer

    @property
    def llm_tokenizer(self):
        """Lazy load the LLM's tokenizer (primarily for fallback counting)."""
        if self._llm_tokenizer is None:
            try:
                 # Use the model name designated for potential local loading
                tokenizer_name = self.config.llm_model_name
                self._llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logger.info(f"Loaded LLM tokenizer for fallback counting: {tokenizer_name}")
            except Exception as e:
                logger.error(f"Failed to load LLM tokenizer {self.config.llm_model_name}. Token counting might be inaccurate. Error: {e}")
                # Provide a dummy tokenizer to avoid crashing, though counts will be wrong
                self._llm_tokenizer = lambda text, **kwargs: {'input_ids': [0] * len(text.split())}
        return self._llm_tokenizer

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using the best available tokenizer."""
        try:
            if isinstance(self.tokenizer, tiktoken.Encoding):
                return len(self.tokenizer.encode(text))
            elif self._llm_tokenizer: # Check if fallback tokenizer is loaded
                # Use the HuggingFace tokenizer's method
                # Use truncation=False and max_length=None explicitly if needed by tokenizer version
                return len(self.llm_tokenizer(text, truncation=False)['input_ids'])
            else:
                 raise RuntimeError("No valid tokenizer available for counting.")
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Approximating using space split.")
            return len(text.split()) # Basic fallback

    # (Chunking methods: _split_fixed_size, _split_paragraph_sentence - unchanged)
    def split_into_chunks(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into chunks based on the configured strategy."""
        if not text:
            return []
        if metadata is None:
            metadata = {}

        metadata["original_doc_id"] = doc_id # Link chunk back to original document

        if self.config.chunk_strategy == "fixed":
            chunks = self._split_fixed_size(doc_id, text, metadata)
        elif self.config.chunk_strategy == "paragraph_sentence":
            chunks = self._split_paragraph_sentence(doc_id, text, metadata)
        else:
            logger.warning(f"Unknown chunk strategy: {self.config.chunk_strategy}. Defaulting to fixed size.")
            chunks = self._split_fixed_size(doc_id, text, metadata)

        logger.debug(f"Split document {doc_id} into {len(chunks)} chunks.")
        return chunks

    def _split_fixed_size(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into fixed size chunks with overlap (token based)."""
        # Use the primary tokenizer (tiktoken or fallback) for splitting
        tokens = self.tokenizer.encode(text) if hasattr(self.tokenizer, 'encode') else self.llm_tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        start_idx = 0
        chunk_seq = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.config.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode using the same tokenizer used for encoding
            try:
                chunk_text = self.tokenizer.decode(chunk_tokens) if hasattr(self.tokenizer, 'decode') else self.llm_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            except Exception as e:
                 logger.error(f"Failed to decode tokens for chunk {chunk_seq} of doc {doc_id}: {e}. Skipping chunk.")
                 start_idx += self.config.chunk_size - self.config.chunk_overlap
                 continue

            if not chunk_text.strip(): # Avoid empty chunks after decoding/stripping
                 start_idx += self.config.chunk_size - self.config.chunk_overlap
                 continue

            chunk_id = f"{doc_id}_chunk_{chunk_seq}"
            chunk_metadata = {**metadata, "chunk_id": chunk_id, "chunk_sequence": chunk_seq}
            chunks.append(Document(id=chunk_id, text=chunk_text.strip(), metadata=chunk_metadata))
            chunk_seq += 1
            if end_idx == len(tokens):
                break
            # Move window with overlap, ensure start_idx doesn't go backward
            start_idx = max(start_idx + 1, start_idx + self.config.chunk_size - self.config.chunk_overlap)

        return chunks

    def _split_paragraph_sentence(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text by paragraphs, then sentences, respecting chunk size."""
        paragraphs = re.split(r'\n\s*\n+', text) # Split by one or more empty lines
        chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_seq = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # Check if the paragraph itself exceeds chunk size
            if para_tokens > self.config.chunk_size:
                # If current chunk has content, finalize it first
                if current_chunk_text:
                    chunk_id = f"{doc_id}_chunk_{chunk_seq}"
                    chunk_metadata = {**metadata, "chunk_id": chunk_id, "chunk_sequence": chunk_seq}
                    chunks.append(Document(id=chunk_id, text=current_chunk_text, metadata=chunk_metadata))
                    chunk_seq += 1
                    current_chunk_text = ""
                    current_chunk_tokens = 0

                # Split the long paragraph into sentences
                try:
                    sentences = sent_tokenize(para)
                except Exception as e:
                    logger.warning(f"Sentence tokenization failed for a long paragraph in doc {doc_id}: {e}. Treating paragraph as single unit, might lead to truncation.")
                    sentences = [para] # Fallback

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence: continue
                    sentence_tokens = self.count_tokens(sentence)

                    # Handle sentence longer than chunk size
                    if sentence_tokens > self.config.chunk_size:
                        logger.warning(f"Sentence exceeds chunk size ({sentence_tokens} > {self.config.chunk_size}) in doc {doc_id}. Truncating sentence.")
                        # Simple truncation (can be improved)
                        tokens = self.tokenizer.encode(sentence) if hasattr(self.tokenizer, 'encode') else self.llm_tokenizer.encode(sentence, add_special_tokens=False)
                        truncated_tokens = tokens[:self.config.chunk_size]
                        try:
                            sentence = self.tokenizer.decode(truncated_tokens) if hasattr(self.tokenizer, 'decode') else self.llm_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                            sentence_tokens = self.count_tokens(sentence) # Recalculate tokens for truncated sentence
                        except Exception as decode_err:
                             logger.error(f"Failed to decode truncated sentence tokens: {decode_err}. Skipping sentence fragment.")
                             continue # Skip this fragment

                    # If adding sentence overflows current chunk
                    if current_chunk_tokens + sentence_tokens > self.config.chunk_size and current_chunk_text:
                        # Finalize current chunk
                        chunk_id = f"{doc_id}_chunk_{chunk_seq}"
                        chunk_metadata = {**metadata, "chunk_id": chunk_id, "chunk_sequence": chunk_seq}
                        chunks.append(Document(id=chunk_id, text=current_chunk_text, metadata=chunk_metadata))
                        chunk_seq += 1
                        # Start new chunk with overlap (optional, simple reset here)
                        # Implement overlap logic if needed by taking last few sentences/tokens
                        current_chunk_text = sentence
                        current_chunk_tokens = sentence_tokens
                    else:
                        # Add sentence to current chunk
                        current_chunk_text += (" " + sentence) if current_chunk_text else sentence
                        current_chunk_tokens += sentence_tokens

            # Else (paragraph fits or adding it might fit)
            else:
                # If adding this paragraph overflows current chunk
                if current_chunk_tokens + para_tokens > self.config.chunk_size and current_chunk_text:
                     # Finalize current chunk
                     chunk_id = f"{doc_id}_chunk_{chunk_seq}"
                     chunk_metadata = {**metadata, "chunk_id": chunk_id, "chunk_sequence": chunk_seq}
                     chunks.append(Document(id=chunk_id, text=current_chunk_text, metadata=chunk_metadata))
                     chunk_seq += 1
                     # Start new chunk with the paragraph
                     current_chunk_text = para
                     current_chunk_tokens = para_tokens
                else:
                     # Add paragraph to current chunk
                     current_chunk_text += ("\n\n" + para) if current_chunk_text else para # Preserve paragraph break
                     current_chunk_tokens += para_tokens


        # Add the last remaining chunk
        if current_chunk_text:
            chunk_id = f"{doc_id}_chunk_{chunk_seq}"
            chunk_metadata = {**metadata, "chunk_id": chunk_id, "chunk_sequence": chunk_seq}
            chunks.append(Document(id=chunk_id, text=current_chunk_text, metadata=chunk_metadata))

        return chunks

    def create_parent_child_chunks(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Document], Dict[str, str]]:
        """
        Create parent-child hierarchy for better retrieval + context.
        
        Strategy:
        - Large parent chunks (1000 tokens) = full context for LLM
        - Small child chunks (256 tokens) = precise retrieval
        
        Returns:
            Tuple of (child_chunks, parent_map)
            - child_chunks: Small chunks to store in vector DB
            - parent_map: Dict mapping child_id -> parent_content
        """
        if not text:
            return [], {}
        
        metadata = metadata or {}
        metadata["original_doc_id"] = doc_id
        
        # STEP 1: Create LARGE parent chunks (1000 tokens, 200 overlap)
        parent_size = 1000
        parent_overlap = 200
        
        # Tokenize entire text
        tokens = self.tokenizer.encode(text) if hasattr(self.tokenizer, 'encode') else self.llm_tokenizer.encode(text, add_special_tokens=False)
        
        parent_chunks = []
        parent_texts = []
        parent_id = 0
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + parent_size, len(tokens))
            parent_tokens = tokens[start_idx:end_idx]
            
            # Decode parent chunk
            try:
                parent_text = self.tokenizer.decode(parent_tokens) if hasattr(self.tokenizer, 'decode') else self.llm_tokenizer.decode(parent_tokens, skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Failed to decode parent chunk {parent_id}: {e}")
                start_idx += parent_size - parent_overlap
                continue
            
            if parent_text.strip():
                parent_texts.append(parent_text)
                parent_chunks.append({
                    'id': f"{doc_id}_parent_{parent_id}",
                    'content': parent_text
                })
                parent_id += 1
            
            # Move to next parent with overlap
            start_idx += parent_size - parent_overlap
        
        logger.info(f"Created {len(parent_chunks)} parent chunks (1000 tokens each) for doc {doc_id}")
        
        # STEP 2: Create SMALL child chunks (256 tokens, 64 overlap) from EACH parent
        child_chunks = []
        parent_map = {}  # child_id -> parent_content
        
        child_size = 256
        child_overlap = 64
        
        for parent_idx, parent_dict in enumerate(parent_chunks):
            parent_text = parent_dict['content']
            parent_chunk_id = parent_dict['id']
            
            # Tokenize this parent
            parent_tokens = self.tokenizer.encode(parent_text) if hasattr(self.tokenizer, 'encode') else self.llm_tokenizer.encode(parent_text, add_special_tokens=False)
            
            # Split parent into children
            child_idx = 0
            start_idx = 0
            
            while start_idx < len(parent_tokens):
                end_idx = min(start_idx + child_size, len(parent_tokens))
                child_tokens = parent_tokens[start_idx:end_idx]
                
                # Decode child chunk
                try:
                    child_text = self.tokenizer.decode(child_tokens) if hasattr(self.tokenizer, 'decode') else self.llm_tokenizer.decode(child_tokens, skip_special_tokens=True)
                except Exception as e:
                    logger.error(f"Failed to decode child chunk: {e}")
                    start_idx += child_size - child_overlap
                    continue
                
                if child_text.strip():
                    child_id = f"{doc_id}_parent_{parent_idx}_child_{child_idx}"
                    
                    child_metadata = {
                        **metadata,
                        'parent_id': parent_chunk_id,
                        'parent_index': parent_idx,
                        'child_index': child_idx,
                        'chunk_type': 'child'  # Mark as child chunk
                    }
                    
                    child_chunks.append(Document(
                        id=child_id,
                        text=child_text,
                        metadata=child_metadata
                    ))
                    
                    # Map child to its parent content
                    parent_map[child_id] = parent_text
                    
                    child_idx += 1
                
                # Move to next child with overlap
                start_idx += child_size - child_overlap
        
        logger.info(f"Created {len(child_chunks)} child chunks (256 tokens each) from {len(parent_chunks)} parents for doc {doc_id}")
        
        return child_chunks, parent_map


class VectorStore:
    """Vector database for storing and retrieving document chunks."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._reranker = None
        self._bm25 = None
        self._corpus_texts: Dict[str, str] = {} # Use dict for easier ID mapping {doc_id: text}
        self._corpus_ids: List[str] = []   # Keep ordered list for BM25 indexing
        self.parent_map: Dict[str, str] = {}  # Parent-child mapping: child_id -> parent_content

    @property
    def client(self):
        """Lazy initialize ChromaDB client."""
        if self._client is None:
            try:
                # Ensure the directory exists
                os.makedirs(self.config.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.config.persist_directory)
                logger.info(f"ChromaDB client initialized. Persistence path: {self.config.persist_directory}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client at {self.config.persist_directory}: {e}", exc_info=True)
                raise
        return self._client

    @property
    def embedding_function(self):
        """Lazy initialize embedding function - USES LOCAL MODELS."""
        if self._embedding_function is None:
            try:
                # Use SentenceTransformer function - runs LOCALLY, no API calls
                self._embedding_function = SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedding_model_name
                )
                logger.info(f"Local SentenceTransformer embedding initialized: {self.config.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding function {self.config.embedding_model_name}: {e}", exc_info=True)
                raise
        return self._embedding_function

    @property
    def collection(self):
        """Lazy initialize ChromaDB collection."""
        if self._collection is None:
            try:
                # Ensure client and embedding function are initialized first
                client = self.client
                emb_fn = self.embedding_function
                self._collection = client.get_or_create_collection(
                    name=self.config.collection_name,
                    embedding_function=emb_fn,
                    metadata={"hnsw:space": "cosine"} # Use cosine distance
                )
                logger.info(f"ChromaDB collection '{self.config.collection_name}' ensured/loaded with {self._collection.count()} documents.")
                # Load existing data for BM25 if collection existed and hybrid search is enabled
                if self._collection.count() > 0 and self.config.use_hybrid_search:
                    self._load_corpus_for_bm25()
            except Exception as e:
                logger.error(f"Failed to get or create ChromaDB collection '{self.config.collection_name}': {e}", exc_info=True)
                raise
        return self._collection

    @property
    def reranker(self):
        """Lazy initialize the CrossEncoder model for reranking."""
        if self._reranker is None and self.config.use_reranking:
            try:
                device = 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
                self._reranker = CrossEncoder(self.config.reranker_model_name, device=device, max_length=512) # Specify max_length
                logger.info(f"CrossEncoder reranker initialized: {self.config.reranker_model_name} on device: {device}")
            except Exception as e:
                logger.error(f"Failed to initialize CrossEncoder {self.config.reranker_model_name}: {e}", exc_info=True)
                # Disable reranking if model fails to load
                logger.warning("Disabling reranking due to initialization error.")
                self.config.use_reranking = False
                self._reranker = None
        return self._reranker

    def _rebuild_bm25_index(self):
        """Rebuilds the BM25 index from the current corpus."""
        if not self.config.use_hybrid_search:
            self._bm25 = None
            return

        if not self._corpus_ids:
             logger.warning("Cannot build BM25 index: corpus is empty.")
             self._bm25 = None
             return

        logger.info(f"Building BM25 index with {len(self._corpus_ids)} documents...")
        try:
            # Ensure texts are fetched for all IDs in the ordered list
            ordered_texts = [self._corpus_texts.get(doc_id, "") for doc_id in self._corpus_ids]
            tokenized_corpus = [text.lower().split() for text in ordered_texts]

            if not any(tokenized_corpus): # Check if all documents are empty after tokenization
                 logger.warning("BM25 index cannot be built: All documents appear empty after tokenization.")
                 self._bm25 = None
                 return

            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 index built successfully.")
        except Exception as e:
             logger.error(f"Failed to build BM25 index: {e}", exc_info=True)
             self._bm25 = None # Disable BM25 on error


    def _load_corpus_for_bm25(self):
        """Load existing documents from ChromaDB to initialize BM25."""
        if not self.config.use_hybrid_search:
            logger.debug("Hybrid search disabled, skipping BM25 corpus load.")
            return

        if not self._collection:
            logger.warning("Collection not initialized, cannot load corpus for BM25.")
            return

        try:
            count = self._collection.count()
            if count == 0:
                logger.info("Collection is empty, BM25 index will be built as documents are added.")
                self._corpus_ids = []
                self._corpus_texts = {}
                self._bm25 = None
                return

            logger.info(f"Loading {count} documents from collection '{self.config.collection_name}' for BM25 index...")
            # Fetch documents in batches
            batch_size = 500 # Adjust batch size based on memory constraints
            all_ids = []
            all_texts = {} # Use dict temporarily

            for offset in tqdm(range(0, count, batch_size), desc="Loading corpus for BM25"):
                try:
                    batch = self._collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=['documents'] # Only need documents for BM25 text
                    )
                    if batch and batch.get('ids') and batch.get('documents'):
                        for i, doc_id in enumerate(batch['ids']):
                             if doc_id not in all_texts: # Avoid duplicates if any
                                all_ids.append(doc_id)
                                all_texts[doc_id] = batch['documents'][i]
                    else:
                        logger.warning(f"Received empty or incomplete batch from ChromaDB at offset {offset}. Stopping corpus load.")
                        break
                except Exception as batch_error:
                     logger.error(f"Error fetching batch from ChromaDB at offset {offset}: {batch_error}", exc_info=True)
                     # Decide whether to continue or stop
                     break # Stop loading on error for safety


            self._corpus_ids = all_ids # Store the order in which they were retrieved/processed
            self._corpus_texts = all_texts # Store the ID->text mapping

            if not self._corpus_ids:
                 logger.warning("No documents successfully loaded from ChromaDB for BM25 index.")
                 self._bm25 = None
                 return

            # Build the index after loading
            self._rebuild_bm25_index()

        except Exception as e:
            logger.error(f"Error loading corpus for BM25 from ChromaDB: {e}", exc_info=True)
            self._corpus_ids = []
            self._corpus_texts = {}
            self._bm25 = None # Ensure BM25 is disabled on error

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """Add documents to the vector store and update BM25 index."""
        if not documents:
            return

        collection = self.collection # Ensure collection is initialized

        num_docs = len(documents)
        added_count = 0
        failed_ids = []
        new_corpus_ids = []
        new_corpus_texts = {}

        for i in tqdm(range(0, num_docs, batch_size), desc="Adding documents to ChromaDB"):
            batch = documents[i:i+batch_size]
            ids = [doc.id for doc in batch]
            texts = [doc.text for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            # Filter out documents already present in the BM25 corpus (by ID) to avoid duplicates in the index
            ids_to_add_chroma = []
            texts_to_add_chroma = []
            metadatas_to_add_chroma = []
            batch_new_corpus_ids = []
            batch_new_corpus_texts = {}

            for j, doc_id in enumerate(ids):
                 ids_to_add_chroma.append(doc_id)
                 texts_to_add_chroma.append(texts[j])
                 metadatas_to_add_chroma.append(metadatas[j])
                 # Check if it's new for BM25
                 if doc_id not in self._corpus_texts:
                     batch_new_corpus_ids.append(doc_id)
                     batch_new_corpus_texts[doc_id] = texts[j]

            if not ids_to_add_chroma:
                 logger.debug(f"Skipping batch starting at index {i}: all documents already in corpus.")
                 continue

            try:
                # Use upsert to add or update documents in ChromaDB
                collection.upsert(
                    ids=ids_to_add_chroma,
                    documents=texts_to_add_chroma,
                    metadatas=metadatas_to_add_chroma
                )
                added_count += len(ids_to_add_chroma)

                # Update BM25 corpus in memory only with *new* documents
                if self.config.use_hybrid_search and batch_new_corpus_ids:
                    new_corpus_ids.extend(batch_new_corpus_ids)
                    new_corpus_texts.update(batch_new_corpus_texts)

            except Exception as e:
                logger.error(f"Failed to add/upsert batch starting at index {i} to ChromaDB: {e}", exc_info=True)
                failed_ids.extend(ids_to_add_chroma)


        logger.info(f"Attempted to add/update {num_docs} documents. Processed: {added_count}. Failed: {len(failed_ids)}.")
        if failed_ids:
             logger.warning(f"Failed document IDs: {failed_ids}")

        # Update the main corpus and rebuild BM25 index if new documents were added
        if self.config.use_hybrid_search and new_corpus_ids:
            logger.info(f"Adding {len(new_corpus_ids)} new documents to BM25 corpus.")
            self._corpus_ids.extend(new_corpus_ids)
            self._corpus_texts.update(new_corpus_texts)
            # Rebuild the index after adding new documents
            self._rebuild_bm25_index()

    def add_documents_with_parent_map(self, documents: List[Document], parent_map: Dict[str, str], batch_size: int = 100) -> None:
        """
        Add child documents to vector store and store parent mapping.
        
        Args:
            documents: List of child chunk Documents (to be stored in ChromaDB)
            parent_map: Dict mapping child_id -> parent_content
            batch_size: Batch size for adding documents
        """
        if not documents:
            return
        
        # Add child chunks to vector store normally
        self.add_documents(documents, batch_size=batch_size)
        
        # Store parent mapping
        self.parent_map.update(parent_map)
        
        logger.info(f"Added {len(documents)} child chunks with {len(set(parent_map.values()))} unique parents")
        logger.info(f"Total parent mappings stored: {len(self.parent_map)}")

    def _chroma_query_to_docs(self, results: Dict[str, Optional[List[Any]]]) -> List[Tuple[Document, float]]:
        """Convert ChromaDB query results to list of (Document, score) tuples."""
        documents = []
        if not results or not results.get('ids') or not results['ids'][0]:
            return []

        ids = results['ids'][0]
        texts = results.get('documents', [None])[0] # Handle missing key
        metadatas = results.get('metadatas', [None])[0] # Handle missing key
        distances = results.get('distances', [None])[0] # Handle missing key

        if texts is None: texts = [None] * len(ids)
        if metadatas is None: metadatas = [{}] * len(ids)
        if distances is None: distances = [1.0] * len(ids) # Default distance if missing

        for i in range(len(ids)):
            # Convert cosine distance to similarity (similarity = 1 - distance)
            # Clamp distance between 0 and 2 for cosine, similarity between -1 and 1
            # Clamp similarity to be >= 0 for practical purposes in RAG scoring? Usually yes.
            similarity = max(0.0, 1.0 - float(distances[i]))

            doc = Document(
                id=ids[i],
                text=texts[i] if texts[i] is not None else "Error: Text not retrieved",
                metadata=metadatas[i] if metadatas[i] is not None else {"Error": "Metadata not retrieved"},
            )
            documents.append((doc, similarity))

        return documents

    def semantic_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Perform semantic search using vector similarity."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count() or top_k), # Avoid querying more than exists
                include=['documents', 'metadatas', 'distances']
            )
            docs_with_scores = self._chroma_query_to_docs(results)
            # Filter based on similarity threshold if needed (though reranking often makes this less critical)
            # filtered_docs = [ (doc, score) for doc, score in docs_with_scores if score >= self.config.similarity_threshold ]
            return docs_with_scores # Return unfiltered for now, reranker will handle relevance

        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            return []

    def keyword_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Perform keyword search using BM25."""
        if not self.config.use_hybrid_search or self._bm25 is None or not self._corpus_ids:
            logger.debug("BM25 search skipped (not enabled, index not ready, or corpus empty).")
            return []

        try:
            query_tokens = query.lower().split()
            # Returns scores for *all* documents in the corpus based on the current index order
            scores = self._bm25.get_scores(query_tokens)

            # Get indices of top-k scores
            num_docs_in_index = len(self._corpus_ids)
            k = min(top_k, num_docs_in_index) # Ensure k is not larger than corpus size
            if k == 0: return []

            # Get indices of documents with scores > 0 (or top k if all are 0)
            positive_score_indices = np.where(scores > 0)[0]
            if len(positive_score_indices) > 0:
                 # Sort only indices with positive scores
                 sorted_positive_indices = positive_score_indices[np.argsort(scores[positive_score_indices])[::-1]]
                 top_indices = sorted_positive_indices[:k]
            else:
                 # If all scores are 0, take the first k indices (arbitrary order)
                 top_indices = np.argsort(scores)[-k:][::-1] # Still sort by score even if 0


            documents = []
            # Normalize scores (optional, simple max scaling for those retrieved)
            max_score = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0

            # Fetch full document info for top results
            top_doc_ids = [self._corpus_ids[idx] for idx in top_indices]

            # Fetch metadatas efficiently in one go if possible
            try:
                chroma_results = self.collection.get(ids=top_doc_ids, include=['metadatas'])
                metadata_map = {chroma_results['ids'][i]: chroma_results['metadatas'][i] for i in range(len(chroma_results['ids']))}
            except Exception as chroma_err:
                 logger.warning(f"Could not fetch metadatas efficiently for BM25 results: {chroma_err}. Fetching one by one (slower).")
                 metadata_map = {} # Fallback below

            for i, idx in enumerate(top_indices):
                doc_id = self._corpus_ids[idx]
                text = self._corpus_texts.get(doc_id, "Error: Text not found in BM25 corpus map")

                # Get metadata (use map or fetch individually as fallback)
                if doc_id in metadata_map:
                     metadata = metadata_map[doc_id]
                else:
                     # Fallback: Fetch individually (slower)
                     try:
                          result = self.collection.get(ids=[doc_id], include=['metadatas'])
                          metadata = result['metadatas'][0] if result and result.get('metadatas') else {}
                     except Exception:
                          metadata = {"warning": "Could not retrieve metadata for BM25 result"}

                normalized_score = scores[idx] / max_score if max_score > 0 else 0.0

                doc = Document(id=doc_id, text=text, metadata=metadata)
                documents.append((doc, normalized_score))

            return documents

        except IndexError as ie:
             logger.error(f"Keyword (BM25) search failed - likely index mismatch. Corpus size: {len(self._corpus_ids)}, BM25 index size may differ. Error: {ie}", exc_info=True)
             # Consider rebuilding the index here if this error occurs frequently
             # self._rebuild_bm25_index()
             return []
        except Exception as e:
            logger.error(f"Keyword (BM25) search failed: {e}", exc_info=True)
            return []


    def hybrid_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Combine results from semantic and keyword search using weighted fusion."""
        if not self.config.use_hybrid_search:
            logger.debug("Hybrid search called but not enabled. Performing semantic search only.")
            # Fetch and return semantic results directly, applying top_k limit
            semantic_results = self.semantic_search(query, top_k)
            return sorted(semantic_results, key=lambda x: x[1], reverse=True)[:top_k]


        # Fetch slightly more results initially if planning to rerank later? No, top_k applies here.
        semantic_results = self.semantic_search(query, top_k)
        keyword_results = self.keyword_search(query, top_k)

        # Combine results using Reciprocal Rank Fusion (RRF) or simple weighted sum
        # Using weighted sum for simplicity: score = alpha * semantic_score + (1-alpha) * keyword_score
        alpha = self.config.hybrid_alpha
        combined_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Process semantic results (score is already similarity 0-1)
        for doc, score in semantic_results:
            combined_scores[doc.id] = combined_scores.get(doc.id, 0.0) + alpha * score
            doc_map[doc.id] = doc

        # Process keyword results (score is normalized BM25 0-1)
        for doc, score in keyword_results:
            combined_scores[doc.id] = combined_scores.get(doc.id, 0.0) + (1.0 - alpha) * score
            # Update doc_map only if the document wasn't seen in semantic results
            # Or decide which version of metadata/text to keep if IDs match
            if doc.id not in doc_map:
                 doc_map[doc.id] = doc


        # Sort combined results by the fused score
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)

        # Create final list of (Document, score) tuples, respecting top_k
        hybrid_results = []
        for doc_id in sorted_ids[:top_k]:
            # Ensure the document exists in our map (it should if ID is in combined_scores)
             if doc_id in doc_map:
                 hybrid_results.append((doc_map[doc_id], combined_scores[doc_id]))
             else:
                  logger.warning(f"Document ID {doc_id} found in combined scores but not in doc_map during hybrid search assembly.")


        logger.debug(f"Hybrid search returned {len(hybrid_results)} documents.")
        return hybrid_results

    def rerank_documents(self, query: str, documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Rerank documents using a CrossEncoder model."""
        if not self.config.use_reranking or self.reranker is None or not documents:
            # If not reranking, still apply the final top_n limit based on original scores
            logger.debug(f"Reranking skipped. Returning top {self.config.rerank_top_n} based on initial scores.")
            return sorted(documents, key=lambda x: x[1], reverse=True)[:self.config.rerank_top_n]

        try:
            # Prepare pairs for the cross-encoder
            pairs = [(query, doc.text) for doc, _ in documents]
            logger.debug(f"Reranking {len(pairs)} pairs...")

            # Predict scores
            scores = self.reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False)

            # Combine documents with new reranker scores
            reranked_docs_with_scores = []
            for i, (doc, initial_score) in enumerate(documents):
                 # Store the reranker score along with the document
                 # Keep original doc object, associate with reranker score
                 reranked_docs_with_scores.append((doc, float(scores[i])))


            # Sort by the new reranker score in descending order
            reranked_docs_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply the final top_n limit after reranking
            final_reranked_docs = reranked_docs_with_scores[:self.config.rerank_top_n]

            if final_reranked_docs:
                 logger.debug(f"Reranked documents returned: {len(final_reranked_docs)}. Top score: {final_reranked_docs[0][1]:.4f}")
            else:
                 logger.debug("Reranking resulted in zero documents after applying top_n limit.")

            return final_reranked_docs

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback: return original documents sorted by initial score, applying top_n
            logger.warning("Falling back to sorting by initial score due to reranking error.")
            return sorted(documents, key=lambda x: x[1], reverse=True)[:self.config.rerank_top_n]


class ResponseGenerator:
    """
    Generates responses using a language model based on retrieved context.
    Supports both local loading via transformers and remote calls via Hugging Face Inference Client.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._local_model = None
        self._local_tokenizer = None
        self._local_pipeline = None
        self._inference_client = None
        self._inference_client_initialized = False # Flag to track initialization attempt

    # --- Properties for Local Model Loading ---
    @property
    def local_tokenizer(self):
        """Lazy load the tokenizer for the local model."""
        if self.config.use_inference_client:
            logger.debug("Using Inference Client, skipping local tokenizer load.")
            return None # Not needed if using inference client
        if self._local_tokenizer is None:
            try:
                self._local_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
                logger.info(f"Local LLM tokenizer loaded: {self.config.llm_model_name}")
            except Exception as e:
                logger.error(f"Failed to load local tokenizer {self.config.llm_model_name}: {e}", exc_info=True)
                # Don't raise, allow None return
        return self._local_tokenizer

    @property
    def local_model(self):
        """Lazy load the local language model."""
        if self.config.use_inference_client:
             logger.debug("Using Inference Client, skipping local model load.")
             return None
        if self._local_model is None:
            try:
                device_map = "auto" if self.config.use_gpu and torch.cuda.is_available() else None
                torch_dtype = torch.float16 if self.config.use_gpu and torch.cuda.is_available() else torch.float32

                # Determine model type
                if "t5" in self.config.llm_model_name.lower() or "bart" in self.config.llm_model_name.lower():
                     ModelClass = AutoModelForSeq2SeqLM
                     logger.info(f"Loading {self.config.llm_model_name} as local Seq2Seq model.")
                else:
                     ModelClass = AutoModelForCausalLM
                     logger.info(f"Loading {self.config.llm_model_name} as local CausalLM model.")

                self._local_model = ModelClass.from_pretrained(
                    self.config.llm_model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    # trust_remote_code=True # Uncomment if model requires it, use with caution
                )
                logger.info(f"Local LLM model loaded: {self.config.llm_model_name} on device_map='{device_map}' with dtype={torch_dtype}")
            except Exception as e:
                logger.error(f"Failed to load local model {self.config.llm_model_name}: {e}", exc_info=True)
                # Don't raise, allow None return
        return self._local_model

    @property
    def local_generation_pipeline(self):
        """Lazy initialize the local generation pipeline."""
        if self.config.use_inference_client:
            # logger.debug("Using Inference Client, skipping local pipeline initialization.")
            return None
        if self._local_pipeline is None:
            # Ensure model and tokenizer are loaded first
            model = self.local_model
            tokenizer = self.local_tokenizer
            if model is None or tokenizer is None:
                logger.error("Cannot initialize local pipeline: Model or Tokenizer failed to load.")
                return None

            try:
                # Determine device for pipeline
                device_num = -1
                if self.config.use_gpu and torch.cuda.is_available():
                     # Let pipeline handle device if model used device_map='auto'
                     # Otherwise, explicitly set to GPU 0 if model is on CPU
                     if model.device.type == 'cuda':
                          device_num = model.device.index if model.device.index is not None else 0
                     elif 'device_map' not in model.config.to_dict(): # Only set if no device_map used
                          device_num = 0


                # --- *** FIX: Determine task based on model type *** ---
                if isinstance(model, AutoModelForSeq2SeqLM):
                    task = "text2text-generation"
                # Check config name as fallback if class check isn't enough (e.g. custom code models)
                elif any(tag in self.config.llm_model_name.lower() for tag in ["t5", "bart", "pegasus"]):
                     task = "text2text-generation"
                else:
                    # Assume CausalLM for others
                    task = "text-generation"
                logger.info(f"Initializing local generation pipeline for task: '{task}'")
                # --- *** END FIX *** ---


                # Handle pipeline arguments carefully
                pipeline_args = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "max_new_tokens": self.config.llm_max_new_tokens,
                }
                if self.config.llm_temperature > 0.01:
                    pipeline_args["temperature"] = self.config.llm_temperature
                    pipeline_args["top_p"] = self.config.llm_top_p
                    pipeline_args["do_sample"] = True
                else:
                    pipeline_args["do_sample"] = False

                # Add device if we determined it and device_map wasn't used
                if device_num != -1 and 'device_map' not in model.config.to_dict():
                    pipeline_args["device"] = device_num
                elif device_num != -1: # Model is on device via device_map, log pipeline device
                     logger.info(f"Model loaded with device_map. Pipeline device will likely be inherited ({model.device}).")
                     # Don't explicitly set pipeline device to potentially avoid conflicts
                else:
                     logger.info("Pipeline running on CPU.")


                self._local_pipeline = pipeline(task, **pipeline_args)
                # Log the actual device the pipeline ended up on
                resolved_device = "unknown"
                if hasattr(self._local_pipeline, 'device') and self._local_pipeline.device:
                     resolved_device = str(self._local_pipeline.device)
                logger.info(f"Local generation pipeline initialized successfully. Task: '{task}'. Device: {resolved_device}")

            except Exception as e:
                logger.error(f"Failed to initialize local generation pipeline: {e}", exc_info=True)
                self._local_pipeline = None # Ensure it's None on failure
        return self._local_pipeline

    # --- ADD this helper method inside ResponseGenerator ---
    def _clean_response(self, raw_response: str, prompt: str) -> str:
        """Removes the prompt from the start of a response if present."""
        # Simple check if response starts with the prompt (potentially after stripping whitespace)
        if raw_response and prompt and raw_response.strip().startswith(prompt.strip()):
            return raw_response[len(prompt):].strip()
        return raw_response.strip()

    # --- Property for Inference Client ---
    @property
    def inference_client(self):
        """Lazy initialize the Hugging Face Inference Client."""
        if not self.config.use_inference_client:
            return None
        if self._inference_client is None and not self._inference_client_initialized:
            self._inference_client_initialized = True # Mark attempt even if it fails
            if not HUGGINGFACE_HUB_AVAILABLE:
                 logger.error("Cannot use Inference Client: 'huggingface_hub' library is not installed.")
                 return None # Client remains None

            token = os.getenv(self.config.hf_api_token_env_var)
            if not token:
                logger.error(f"Cannot use Inference Client: Environment variable '{self.config.hf_api_token_env_var}' not set.")
                return None # Client remains None
            try:
                self._inference_client = InferenceClient(token=token)
                logger.info("Hugging Face Inference Client initialized successfully.")

                # --- Optional: Check model accessibility ---
                model_to_check = self.config.inference_client_model or self.config.llm_model_name
                if not model_to_check:
                     logger.error("No model specified for Inference Client.")
                     self._inference_client = None
                     return None

                logger.info(f"Verifying access to Inference Client model: {model_to_check}")
                try:
                     api = HfApi()
                     api.model_info(model_to_check, token=token)
                     logger.info(f"Inference model '{model_to_check}' found and accessible.")
                except RepositoryNotFoundError:
                     logger.error(f"Inference model '{model_to_check}' not found on Hugging Face Hub.")
                     self._inference_client = None # Invalidate client
                except GatedRepoError: # Handle specific error for gated models
                      logger.warning(f"Inference model '{model_to_check}' is gated. Ensure your token '{self.config.hf_api_token_env_var}' has been granted access.")
                      # Proceed, assuming user has access, but log warning.
                except Exception as model_check_err:
                     # Log other potential errors (network issues, auth problems)
                     logger.warning(f"Could not verify inference model '{model_to_check}' accessibility: {model_check_err}")
                # --- End Optional Check ---

            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face Inference Client: {e}", exc_info=True)
                self._inference_client = None # Ensure it's None on failure

        return self._inference_client

    # --- Formatting and Generation ---

    def check_grounding(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Verify answer faithfulness using context overlap check.
        Returns grounding metrics and filtered answer.
        """
        if not self.config.use_grounding_check:
            return {"grounded": True, "overlap_ratio": 1.0, "filtered_answer": answer}
        
        try:
            # Tokenize answer and context into words
            answer_tokens = set(answer.lower().split())
            context_tokens = set(context.lower().split())
            
            # Remove common stop words for better signal
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
                         'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'should', 'could', 'may', 'might', 'must', 'can'}
            
            answer_tokens_clean = answer_tokens - stop_words
            context_tokens_clean = context_tokens - stop_words
            
            # Calculate overlap ratio
            if len(answer_tokens_clean) == 0:
                overlap_ratio = 0.0
            else:
                overlap = len(answer_tokens_clean & context_tokens_clean)
                overlap_ratio = overlap / len(answer_tokens_clean)
            
            # Check if answer is grounded
            grounded = overlap_ratio >= self.config.min_context_overlap
            
            if not grounded:
                logger.warning(f"Answer failed grounding check. Overlap: {overlap_ratio:.2f}, Threshold: {self.config.min_context_overlap}")
                filtered_answer = f"[Low Confidence] {answer}"
            else:
                filtered_answer = answer
                
            return {
                "grounded": grounded,
                "overlap_ratio": overlap_ratio,
                "filtered_answer": filtered_answer
            }
            
        except Exception as e:
            logger.error(f"Grounding check failed: {e}")
            return {"grounded": True, "overlap_ratio": 0.0, "filtered_answer": answer, "error": str(e)}

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a string context."""
        context_parts = []
        for i, doc in enumerate(documents):
            source_info = f"Source {i+1} (ID: {doc.id})"
            title = doc.metadata.get('title')
            if title:
                source_info += f" Title: {title}"
            # Add text, ensuring it's a string
            doc_text = str(doc.text) if doc.text is not None else ""
            context_parts.append(f"[{source_info}]:\n{doc_text}")
        return "\n\n".join(context_parts)

    def _build_llama3_prompt(self, query: str, context: str) -> str:
         """Builds a prompt string specifically for Llama 3 Instruct."""
         # Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
         # Use a concise system prompt focused on the RAG task
         system_prompt = "You are an AI assistant. Answer the user's question based *only* on the provided context. If the context doesn't contain the necessary information, state that clearly."
         # Structure according to Llama 3 format
         prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
         prompt += f"<|start_header_id|>user<|end_header_id|>\n\n## Context:\n{context}\n\n## Question:\n{query}<|eot_id|>"
         prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n" # Signal for the model to start generating
         return prompt

    def _build_generic_prompt(self, query: str, context: str) -> str:
         """Builds a generic prompt suitable for models like Flan-T5."""
         return f"""Answer the following question based *only* on the provided context. If the context does not provide the information needed, say "I cannot answer the question based on the provided context."

Context:
---
{context}
---

Question: {query}

Answer:"""

    def _post_process_answer(self, answer: str, query: str) -> str:
        """Post-process answer for conciseness to match ground truth format."""
        # Remove common prefixes
        prefixes_to_remove = [
            "The answer is ",
            "Based on the context, ",
            "According to the context, ",
            "From the context, ",
            "The context states that ",
            "Yes, ",
            "No, ",
        ]
        
        processed = answer.strip()
        for prefix in prefixes_to_remove:
            if processed.startswith(prefix):
                processed = processed[len(prefix):]
                break
        
        # For yes/no questions, extract just yes/no
        query_lower = query.lower()
        if any(q in query_lower for q in ["did ", "does ", "do ", "was ", "were ", "is ", "are ", "will ", "would ", "can ", "could "]):
            # Check if answer contains yes/no
            answer_lower = processed.lower()
            if answer_lower.startswith("yes"):
                return "yes"
            elif answer_lower.startswith("no"):
                return "no"
        
        # Take only first sentence for conciseness
        sentences = processed.split(". ")
        if sentences:
            processed = sentences[0]
        
        # Remove trailing punctuation that's not needed
        processed = processed.rstrip(".")
        
        return processed.strip()

    def generate_answer(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate an answer using the configured method (Gemini API, HF Inference Client, or local pipeline)."""
        start_time = time.time()
        context = self.format_context(documents)
        response_text = "Error: Could not generate response."
        generation_metrics = {}
        prompt_used = "" # Store the prompt for debugging/logging

        # Priority 1: Use Gemini API if configured
        if self.config.use_gemini_api:
            if not GEMINI_AVAILABLE:
                logger.error("Gemini API requested but google-generativeai library not available.")
                response_text = "Error: Gemini library not installed."
                generation_metrics = {"error": response_text, "method": "gemini_api"}
            else:
                try:
                    # Get API key from environment
                    api_key = os.getenv(self.config.gemini_api_key_env_var)
                    if not api_key:
                        raise ValueError(f"Gemini API key not found in environment variable: {self.config.gemini_api_key_env_var}")
                    
                    # Configure Gemini
                    genai.configure(api_key=api_key)
                    
                    # Disable safety settings to avoid blocks
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    
                    model = genai.GenerativeModel(
                        self.config.gemini_model_name,
                        safety_settings=safety_settings
                    )
                    
                    # EMERGENCY FIX 2: Better prompt that avoids safety blocks
                    # Truncate context if too long (max 4000 chars)
                    if len(context) > 4000:
                        logger.warning(f"Context too long ({len(context)} chars), truncating to 4000")
                        context = context[:4000] + "\n...(truncated)"
                    
                    user_prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Context Information:
{context}

Question: {query}

Instructions:
- Answer directly and concisely
- Use ONLY information from the context above
- If the context doesn't contain the answer, say "The provided context does not contain enough information to answer this question."
- Do not add external knowledge
- Be factual and specific

Answer:"""
                    
                    logger.info(f"Sending request to Gemini API (model: {self.config.gemini_model_name})")
                    logger.debug(f"Context length: {len(context)} chars, Prompt length: {len(user_prompt)} chars")
                    
                    # EMERGENCY FIX 3: Retry logic with prompt simplification
                    max_retries = 3
                    response_text = None
                    
                    for attempt in range(max_retries):
                        try:
                            # Generate response
                            response = model.generate_content(
                                user_prompt,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=0.1,  # Low temp for factual answers
                                    top_p=0.95,
                                    max_output_tokens=256,
                                )
                            )
                            
                            # Check for blocks
                            if not response.candidates or not response.candidates[0].content.parts:
                                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                                logger.warning(f"Gemini blocked response (attempt {attempt+1}). Finish reason: {finish_reason}")
                                
                                # Simplify prompt and retry
                                if attempt < max_retries - 1:
                                    # Remove formatting and shorten
                                    context = context[:2000]
                                    user_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer using only the context:"
                                    continue
                                else:
                                    response_text = f"Unable to generate answer (model blocked after {max_retries} attempts)"
                                    break
                            else:
                                response_text = response.text.strip()
                                logger.info(f"Gemini generated answer successfully (attempt {attempt+1})")
                                break
                                
                        except Exception as retry_error:
                            logger.error(f"Generation attempt {attempt+1} failed: {retry_error}")
                            if attempt == max_retries - 1:
                                response_text = f"Generation failed after {max_retries} attempts: {str(retry_error)}"
                    
                    if response_text is None:
                        response_text = "Error: No response generated"
                    generation_time = time.time() - start_time
                    
                    logger.info(f"Gemini API generation took {generation_time:.2f} seconds.")
                    generation_metrics = {
                        "generation_time_sec": generation_time,
                        "method": "gemini_api",
                        "model_used": self.config.gemini_model_name
                    }
                    
                except Exception as e:
                    logger.error(f"Gemini API call failed: {e}", exc_info=True)
                    response_text = f"Error: Failed to get response from Gemini API. Details: {e}"
                    generation_metrics = {"error": str(e), "method": "gemini_api"}
                    
        # Priority 1.5: Use OpenRouter API if configured (RECOMMENDED - best for RAG!)
        elif self.config.use_openrouter_api:
            try:
                import requests
                
                # Get API key from environment
                api_key = os.getenv(self.config.openrouter_api_key_env_var)
                if not api_key:
                    raise ValueError(f"OpenRouter API key not found in environment variable: {self.config.openrouter_api_key_env_var}")
                
                # CRITICAL: Concise prompt for short answers
                if len(context) > 5000:  # Increased from 3000 - allow more context
                    logger.warning(f"Context too long ({len(context)} chars), truncating to 5000")
                    context = context[:5000]
                
                # IMPROVED: Smart prompt with reasoning for ambiguous contexts
                user_prompt = f"""Context:
{context}

Question: {query}

Instructions: Answer concisely based on the context. Use logical reasoning if needed.

IMPORTANT RULES:
1. For yes/no questions: Answer "yes" or "no" ONLY
2. For factual questions: Give the specific answer (name, title, number, etc.)
3. If multiple interpretations exist, use the most relevant one
4. You MAY infer from context (e.g., if context mentions "X is an American film about X", then X (the person) was likely American)
5. ONLY say "Unknown" if absolutely no relevant info exists

Think step-by-step but answer in 1-3 words maximum.

Answer:"""
                
                logger.info(f"Sending request to OpenRouter API (model: {self.config.openrouter_model_name})")
                logger.debug(f"Context length: {len(context)} chars")
                
                # OpenRouter API call
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.openrouter_model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that answers questions concisely using only the provided context."
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        "max_tokens": self.config.llm_max_new_tokens,
                        "temperature": self.config.llm_temperature,
                        "top_p": self.config.llm_top_p,
                        "stop": ["\n\n", "Question:", "Context:"],  # Stop early for conciseness
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content'].strip()
                    
                    # Post-process for conciseness
                    response_text = self._post_process_answer(response_text, query)
                    
                    logger.info(f"OpenRouter generated answer: {response_text}")
                else:
                    error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    response_text = f"Error: {error_msg}"
                
                generation_time = time.time() - start_time
                generation_metrics = {
                    "generation_time_sec": generation_time,
                    "method": "openrouter_api",
                    "model_used": self.config.openrouter_model_name
                }
                
            except Exception as e:
                logger.error(f"OpenRouter API call failed: {e}", exc_info=True)
                response_text = f"Error: Failed to get response from OpenRouter API. Details: {e}"
                generation_metrics = {"error": str(e), "method": "openrouter_api"}
                    
        # Priority 2: Use HuggingFace Inference Client if configured
        elif self.config.use_inference_client:
            # --- Use Inference Client ---
            client = self.inference_client # Trigger lazy loading/check
            if client:
                model_id = self.config.inference_client_model or self.config.llm_model_name
                if not model_id:
                     response_text = "Error: No model ID specified for Inference Client."
                     generation_metrics = {"error": response_text, "method": "inference_client"}
                     model_id = "N/A" # Placeholder for logging
                else:
                    # Determine if we should use chat completion or text generation
                    use_chat_api = "llama" in model_id.lower() or "instruct" in model_id.lower() or "chat" in model_id.lower()
                    
                    try:
                        logger.info(f"Sending request to Inference API (model: {model_id})")
                        
                        if use_chat_api:
                            # Use chat completion API for chat/instruct models
                            logger.debug(f"Using chat completion API for model {model_id}")
                            
                            # Build messages for chat API
                            system_prompt = "You are a helpful assistant that answers questions based on the provided context. Provide concise, accurate answers."
                            user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a concise answer based on the context above."
                            
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message}
                            ]
                            
                            api_response = client.chat_completion(
                                model=model_id,
                                messages=messages,
                                max_tokens=self.config.llm_max_new_tokens,
                                temperature=self.config.llm_temperature if self.config.llm_temperature > 0.01 else 0.1,
                                top_p=self.config.llm_top_p if self.config.llm_temperature > 0.01 else None,
                            )
                            
                            # Extract response from chat completion
                            if hasattr(api_response, 'choices') and len(api_response.choices) > 0:
                                response_text = api_response.choices[0].message.content
                            elif isinstance(api_response, dict) and 'choices' in api_response:
                                response_text = api_response['choices'][0]['message']['content']
                            else:
                                logger.warning(f"Unexpected chat completion response format: {type(api_response)}")
                                response_text = "Error: Unexpected response format from chat API."
                        else:
                            # Use text generation API for base models
                            logger.debug(f"Using text generation API for model {model_id}")
                            prompt_used = self._build_generic_prompt(query, context)
                            
                            api_response = client.text_generation(
                                model=model_id,
                                prompt=prompt_used,
                                max_new_tokens=self.config.llm_max_new_tokens,
                                temperature=self.config.llm_temperature if self.config.llm_temperature > 0.01 else None,
                                top_p=self.config.llm_top_p if self.config.llm_temperature > 0.01 else None,
                                do_sample=True if self.config.llm_temperature > 0.01 else False,
                                return_full_text=False
                            )

                            # Extract response text from API response
                            if isinstance(api_response, str):
                                response_text = api_response
                            elif hasattr(api_response, 'generated_text'):
                                response_text = api_response.generated_text
                            elif isinstance(api_response, dict) and 'generated_text' in api_response:
                                response_text = api_response['generated_text']
                            else:
                                logger.warning(f"Unexpected response type ({type(api_response)})")
                                response_text = "Error: Received unexpected or empty response from API."


                    except Exception as e:
                        logger.error(f"Inference API call failed: {e}", exc_info=True)
                        response_text = f"Error: Failed to get response from Inference API. Details: {e}"
                        generation_metrics = {"error": str(e)} # Add error to metrics

                generation_time = time.time() - start_time
                logger.info(f"Inference API generation took {generation_time:.2f} seconds.")
                generation_metrics.update({ # Merge metrics
                    "generation_time_sec": generation_time,
                    "method": "inference_client",
                    "model_used": model_id
                    # Token counts from API are not standard, omit for now
                })

            else:
                response_text = "Error: Inference Client is enabled but could not be initialized. Check logs and HF token."
                generation_metrics = {"error": "Inference Client initialization failed", "method": "inference_client"}

        else:
            # --- Use Local Pipeline ---
            pipe = self.local_generation_pipeline # Trigger lazy loading/check
            if pipe:
                # Typically use generic prompt for local models unless specific format known
                prompt_used = self._build_generic_prompt(query, context)
                logger.info(f"Running local generation pipeline (model: {self.config.llm_model_name}). Prompt length: {len(prompt_used)} chars.")
                try:
                    results = pipe(prompt_used, num_return_sequences=1)

                    # Process local pipeline results (same logic as before)
                    if isinstance(results, list) and results:
                         if isinstance(results[0], dict):
                              generated_text_key = 'generated_text' if 'generated_text' in results[0] else \
                                                   'summary_text' if 'summary_text' in results[0] else None
                              if generated_text_key:
                                   raw_response = results[0][generated_text_key]
                                   # Remove prompt for CausalLM if return_full_text=True implicitly
                                   if isinstance(self.local_model, AutoModelForCausalLM) and raw_response.startswith(prompt_used):
                                        response_text = raw_response[len(prompt_used):].strip()
                                   else:
                                        response_text = raw_response.strip() # Assume Seq2Seq or return_full_text=False
                              else:
                                   logger.warning(f"Could not find standard text key in local pipeline output: {results[0].keys()}")
                                   response_text = str(results[0]) # Fallback
                         else: response_text = str(results[0])
                    elif isinstance(results, str): response_text = results
                    else:
                         logger.warning(f"Unexpected output format from local pipeline: {type(results)}")
                         response_text = "Error: Could not parse local LLM response."

                    generation_time = time.time() - start_time
                    logger.info(f"Local pipeline generation took {generation_time:.2f} seconds.")
                    # Token counting can be added here using self.local_tokenizer if needed
                    generation_metrics = {
                        "generation_time_sec": generation_time,
                        "method": "local_pipeline",
                        "model_used": self.config.llm_model_name
                    }

                except Exception as e:
                    logger.error(f"Local pipeline generation failed: {e}", exc_info=True)
                    response_text = "Error: Failed to generate response from local LLM."
                    generation_metrics = {"error": str(e)}
            else:
                response_text = "Error: Local generation pipeline not available. Check model loading logs."
                generation_metrics = {"error": "Local pipeline initialization failed", "method": "local_pipeline"}

        # Clean up response (remove potential leftover stop sequences if API/model didn't handle it)
        if self.config.use_inference_client and "llama-3" in (self.config.inference_client_model or ""):
             response_text = response_text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

        # Apply grounding check if enabled
        grounding_result = self.check_grounding(response_text, context)
        response_text = grounding_result["filtered_answer"]
        generation_metrics["grounding_check"] = {
            "grounded": grounding_result["grounded"],
            "overlap_ratio": grounding_result["overlap_ratio"]
        }

        # --- Final Result Structure ---
        return {
            "answer": response_text.strip(),
            "sources": [{"id": doc.id, "text_preview": str(doc.text[:100])+"...", "metadata": doc.metadata} for doc in documents],
            "metrics": generation_metrics,
            "_debug_prompt": prompt_used # Include prompt for debugging if needed
        }


# --- Evaluation ---

class RAGEvaluator:
    """Evaluates the performance of the RAG system components."""

    def __init__(self):
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = Rouge()
                logger.info("ROUGE scorer initialized for evaluation.")
            except Exception as e:
                logger.error(f"Failed to initialize ROUGE scorer: {e}")
                self.rouge_scorer = None
        else:
             logger.warning("ROUGE library not available. Generation evaluation will be limited.")

    def evaluate_retrieval(self, retrieved_docs: List[Document], relevant_doc_ids: List[str]) -> Dict[str, float]:
        """Evaluate retrieval performance (Precision, Recall, F1)."""
        # (Unchanged evaluation logic)
        if not retrieved_docs and not relevant_doc_ids:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not relevant_doc_ids:
             # If no relevant docs, precision is 0 if anything was retrieved, 1 if nothing was retrieved. Recall/F1 undefined or 0.
            return {"precision": 0.0 if retrieved_docs else 1.0, "recall": 0.0, "f1": 0.0}
        if not retrieved_docs:
             return {"precision": 0.0, "recall": 0.0, "f1": 0.0} # Nothing retrieved fails to recall relevant docs

        retrieved_ids = {doc.id for doc in retrieved_docs}
        relevant_ids_set = set(relevant_doc_ids)

        true_positives = len(retrieved_ids.intersection(relevant_ids_set))
        retrieved_count = len(retrieved_ids)
        relevant_count = len(relevant_ids_set)

        precision = true_positives / retrieved_count if retrieved_count > 0 else 0.0
        recall = true_positives / relevant_count # relevant_count > 0 asserted above
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}


    def evaluate_generation(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Evaluate generated answer quality using ROUGE."""
        # (Unchanged evaluation logic)
        if self.rouge_scorer is None:
            logger.warning("ROUGE scorer not available, skipping generation evaluation.")
            return {"rouge-1": np.nan, "rouge-2": np.nan, "rouge-l": np.nan, "error": "ROUGE not installed"}

        # Ensure inputs are strings and handle potential None values
        gen_ans = str(generated_answer) if generated_answer is not None else ""
        ref_ans = str(reference_answer) if reference_answer is not None else ""

        if not gen_ans or not ref_ans:
            # Return 0 if either is empty, as ROUGE calculation would fail or be meaningless
            # Log if only one is empty?
            if not gen_ans: logger.debug("Generated answer is empty for ROUGE.")
            if not ref_ans: logger.debug("Reference answer is empty for ROUGE.")
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

        try:
            scores = self.rouge_scorer.get_scores(gen_ans, ref_ans)[0]
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except ValueError as ve:
             logger.warning(f"ROUGE calculation ValueError (likely empty hypothesis/reference after processing): {ve}. Generated: '{gen_ans[:50]}...', Reference: '{ref_ans[:50]}...'")
             return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {"rouge-1": np.nan, "rouge-2": np.nan, "rouge-l": np.nan, "error": str(e)}


# --- Parallel Retrieval Fusion ---

@dataclass
class RetrievalResult:
    """Single retrieval strategy result"""
    strategy_name: str
    documents: List[Document]
    scores: List[float]
    latency_ms: float


class ParallelRetrievalFusion:
    """
    Execute multiple retrieval strategies in parallel and fuse results using RRF
    NO LLM CALLS - pure retrieval optimization!
    """
    
    def __init__(self, rag_system):
        self.rag = rag_system
        self.vector_store = rag_system.vector_store
    
    async def retrieve_parallel(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Main entry point: Run all strategies in parallel, then fuse
        """
        import time
        start = time.time()
        
        # Launch all strategies in parallel
        tasks = [
            self._strategy_semantic(query, top_k * 2),
            self._strategy_bm25(query, top_k * 2),
            self._strategy_hybrid(query, top_k * 2),
            self._strategy_expanded_query(query, top_k * 2),
            self._strategy_entity_focused(query, top_k * 2),
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        valid_results = [r for r in results if isinstance(r, RetrievalResult)]
        
        if not valid_results:
            logger.error("All retrieval strategies failed!")
            return []
        
        logger.info(f"Parallel retrieval: {len(valid_results)} strategies in {(time.time()-start)*1000:.1f}ms")
        
        # Fuse results using Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(valid_results, top_k=top_k)
        
        return fused_docs
    
    # ==================== RETRIEVAL STRATEGIES ====================
    
    async def _strategy_semantic(self, query: str, top_k: int) -> RetrievalResult:
        """Pure semantic/vector search"""
        import time
        start = time.time()
        
        try:
            # Call semantic search from vector store
            results = await asyncio.to_thread(
                self.vector_store.collection.query,
                query_texts=[query],
                n_results=top_k
            )
            
            docs = []
            scores = []
            if results and results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0.0] * len(results['documents'][0])
                )):
                    doc_id = results['ids'][0][i] if results['ids'] else f"doc_{i}"
                    score = 1.0 - distance  # Convert distance to similarity
                    
                    doc = Document(id=doc_id, text=doc_text, metadata=metadata)
                    doc.score = score
                    docs.append(doc)
                    scores.append(score)
            
            return RetrievalResult(
                strategy_name="semantic",
                documents=docs,
                scores=scores,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def _strategy_bm25(self, query: str, top_k: int) -> RetrievalResult:
        """Pure BM25 keyword search"""
        import time
        start = time.time()
        
        try:
            # Use existing keyword_search method
            results = await asyncio.to_thread(
                self.vector_store.keyword_search,
                query,
                top_k
            )
            
            docs = []
            scores = []
            for doc, score in results:
                doc.score = score
                docs.append(doc)
                scores.append(score)
            
            return RetrievalResult(
                strategy_name="bm25",
                documents=docs,
                scores=scores,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return RetrievalResult("bm25", [], [], 0.0)
    
    async def _strategy_hybrid(self, query: str, top_k: int) -> RetrievalResult:
        """Hybrid search (semantic + BM25)"""
        import time
        start = time.time()
        
        try:
            # Use existing hybrid_search method
            results = await asyncio.to_thread(
                self.vector_store.hybrid_search,
                query,
                top_k
            )
            
            docs = []
            scores = []
            for doc, score in results:
                doc.score = score
                docs.append(doc)
                scores.append(score)
            
            return RetrievalResult(
                strategy_name="hybrid",
                documents=docs,
                scores=scores,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return RetrievalResult("hybrid", [], [], 0.0)
    
    async def _strategy_expanded_query(self, query: str, top_k: int) -> RetrievalResult:
        """Query expansion strategy"""
        import time
        start = time.time()
        
        try:
            # Simple query expansion without LLM
            expanded = self._expand_query_simple(query)
            
            # Use semantic search on expanded query
            results = await asyncio.to_thread(
                self.vector_store.collection.query,
                query_texts=[expanded],
                n_results=top_k
            )
            
            docs = []
            scores = []
            if results and results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0.0] * len(results['documents'][0])
                )):
                    doc_id = results['ids'][0][i] if results['ids'] else f"doc_{i}"
                    score = 1.0 - distance
                    
                    doc = Document(id=doc_id, text=doc_text, metadata=metadata)
                    doc.score = score
                    docs.append(doc)
                    scores.append(score)
            
            return RetrievalResult(
                strategy_name="expanded",
                documents=docs,
                scores=scores,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            logger.error(f"Expanded query search failed: {e}")
            raise
    
    async def _strategy_entity_focused(self, query: str, top_k: int) -> RetrievalResult:
        """Focus on entities/proper nouns in query"""
        import time
        import re
        start = time.time()
        
        try:
            # Extract entities (proper nouns)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            entity_query = ' '.join(entities) if entities else query
            
            # Use existing keyword_search with entity query
            results = await asyncio.to_thread(
                self.vector_store.keyword_search,
                entity_query,
                top_k
            )
            
            docs = []
            scores = []
            for doc, score in results:
                doc.score = score
                docs.append(doc)
                scores.append(score)
            
            return RetrievalResult(
                strategy_name="entity_focused",
                documents=docs,
                scores=scores,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            logger.error(f"Entity-focused search failed: {e}")
            return RetrievalResult("entity_focused", [], [], 0.0)
    
    # ==================== FUSION ALGORITHMS ====================
    
    def _reciprocal_rank_fusion(self, results: List[RetrievalResult], 
                                 top_k: int = 10, k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) - industry standard
        Score = sum(1 / (k + rank)) across all strategies
        """
        doc_scores = {}
        doc_objects = {}
        
        for result in results:
            strategy_weight = self._get_strategy_weight(result.strategy_name)
            
            for rank, doc in enumerate(result.documents, start=1):
                # RRF score
                rrf_score = strategy_weight / (k + rank)
                
                if doc.id in doc_scores:
                    doc_scores[doc.id] += rrf_score
                else:
                    doc_scores[doc.id] = rrf_score
                    doc_objects[doc.id] = doc
        
        # Sort by fused scores
        sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build result list
        fused_docs = []
        for doc_id, score in sorted_ids:
            doc = doc_objects[doc_id]
            doc.score = score
            fused_docs.append(doc)
        
        logger.info(f"RRF fusion: {len(doc_scores)} unique docs -> top {len(fused_docs)}")
        return fused_docs
    
    def _get_strategy_weight(self, strategy_name: str) -> float:
        """Weight different strategies based on performance"""
        weights = {
            "semantic": 1.2,
            "bm25": 0.8,
            "hybrid": 1.5,
            "expanded": 1.0,
            "entity_focused": 0.9
        }
        return weights.get(strategy_name, 1.0)
    
    def _expand_query_simple(self, query: str) -> str:
        """Simple query expansion without LLM"""
        synonyms = {
            "born": ["birth", "birthplace"],
            "director": ["directed", "filmmaker"],
            "wrote": ["written", "author", "writer"],
            "created": ["creator", "founded", "established"],
            "nationality": ["born", "from", "citizen"],
            "same": ["identical", "equal", "both"]
        }
        
        words = query.lower().split()
        expanded_terms = []
        
        for word in words:
            expanded_terms.append(word)
            if word in synonyms:
                expanded_terms.extend(synonyms[word])
        
        return ' '.join(expanded_terms)


# --- Main RAG Class ---

class AdvancedRAG:
    """Orchestrates the Advanced RAG system using modular components."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        logger.info(f"Initializing AdvancedRAG with config: {self.config}")

        self.text_processor = TextProcessor(self.config)
        self.vector_store = VectorStore(self.config) # Handles Chroma, BM25, Reranker
        self.response_generator = ResponseGenerator(self.config) # Handles Local LLM or API LLM
        self.evaluator = RAGEvaluator()
        
        # Parallel Retrieval Fusion system
        self.parallel_retriever = ParallelRetrievalFusion(self)
        logger.info("Parallel Retrieval Fusion initialized")

        # LRU cache applied directly to the retrieve method

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> List[str]:
        """Processes, chunks, and adds a single document to the vector store."""
        # (Unchanged logic)
        if not text or not isinstance(text, str):
            logger.warning(f"Attempted to add an empty or non-string document (type: {type(text)}). Skipping.")
            return []
        metadata = metadata or {}
        # Generate ID based on text hash and timestamp if not provided
        doc_id = doc_id or f"doc_{hashlib.md5(text[:1000].encode()).hexdigest()}_{int(time.time())}"

        # Add ingestion timestamp
        metadata["ingested_at"] = datetime.now().isoformat()
        metadata.setdefault("source_document_id", doc_id) # Track original ID

        try:
            # 1. Chunk the document (with parent-child if enabled)
            if self.config.use_parent_child_chunking:
                # Use parent-child hierarchical chunking
                chunks, parent_map = self.text_processor.create_parent_child_chunks(doc_id, text, metadata)
                if not chunks:
                    logger.warning(f"Document {doc_id} resulted in no child chunks after parent-child processing.")
                    return []
                
                # Add child chunks with parent mapping
                self.vector_store.add_documents_with_parent_map(chunks, parent_map)
                logger.info(f"Added document {doc_id} using parent-child chunking: {len(chunks)} children, {len(set(parent_map.values()))} parents")
            else:
                # Use standard chunking
                chunks = self.text_processor.split_into_chunks(doc_id, text, metadata)
                if not chunks:
                    logger.warning(f"Document {doc_id} resulted in no chunks after processing.")
                    return []

                # Add chunks to vector store (handles Chroma and BM25 update)
                self.vector_store.add_documents(chunks)

            return [chunk.id for chunk in chunks]

        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}", exc_info=True)
            return []

    def add_documents_from_dataframe(self, df: pd.DataFrame, text_column: str, metadata_columns: Optional[List[str]] = None, id_column: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Adds documents from a pandas DataFrame."""
        # (Unchanged logic)
        all_added_chunk_ids = []
        all_failed_doc_ids = []
        metadata_columns = metadata_columns or []

        logger.info(f"Adding documents from DataFrame with {len(df)} rows. Text: '{text_column}', Metadata: {metadata_columns}, ID: '{id_column}'.")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing DataFrame rows"):
            text = row.get(text_column)
            # Validate text content
            if not text or pd.isna(text) or not isinstance(text, str) or not text.strip():
                original_id = str(row[id_column]) if id_column and id_column in row else f"df_index_{index}"
                logger.warning(f"Skipping row {index} (ID: {original_id}) due to missing, empty, or non-string text in column '{text_column}'.")
                if original_id: all_failed_doc_ids.append(original_id)
                continue

            doc_id = str(row[id_column]) if id_column and id_column in row and not pd.isna(row[id_column]) else None
            metadata = {col: row[col] for col in metadata_columns if col in row and not pd.isna(row[col])}

            # Add original DataFrame index to metadata
            metadata["dataframe_index"] = index

            # Use provided ID or generate one including index
            doc_id_to_use = doc_id or f"df_{index}_{hashlib.md5(str(text[:100]).encode()).hexdigest()}"
            metadata.setdefault("source_document_id", doc_id_to_use) # Ensure source ID is set

            chunk_ids = self.add_document(str(text), metadata, doc_id=doc_id_to_use)
            if chunk_ids:
                all_added_chunk_ids.extend(chunk_ids)
            else:
                # If add_document failed, record the intended doc ID
                all_failed_doc_ids.append(doc_id_to_use)


        logger.info(f"Finished adding documents from DataFrame. Added {len(all_added_chunk_ids)} chunks. Failed to process {len(all_failed_doc_ids)} documents.")
        return all_added_chunk_ids, all_failed_doc_ids

    # Cache retrieval results based on query and relevant config parameters
    @lru_cache(maxsize=1024) # Adjust cache size as needed
    def _cached_retrieve_rerank(self, query: str, k: int, use_hybrid: bool, hybrid_alpha: float, use_rerank: bool, rerank_n: int) -> List[Tuple[Document, float]]:
        """Internal cached retrieval & reranking function."""
        logger.debug(f"Cache miss/lookup for query: '{query[:50]}...' (k={k}, hybrid={use_hybrid}, rerank={use_rerank}, rerank_n={rerank_n})")
        # 1. Initial Retrieval (Hybrid or Semantic)
        if use_hybrid:
            # We pass k here, as hybrid search combines results before returning top_k
            retrieved_docs_with_scores = self.vector_store.hybrid_search(query, k)
        else:
            retrieved_docs_with_scores = self.vector_store.semantic_search(query, k)

        # 2. Reranking (if enabled)
        if use_rerank:
            # Rerank the initially retrieved docs and apply the final top_n limit
            reranked_docs_with_scores = self.vector_store.rerank_documents(query, retrieved_docs_with_scores)
            # Return the reranked results (already limited to rerank_n)
            return reranked_docs_with_scores
        else:
            # If not reranking, return the initial results, sorted by score and limited to k (or maybe rerank_n? Let's stick to k for consistency)
            return sorted(retrieved_docs_with_scores, key=lambda x: x[1], reverse=True)[:k]


    def retrieve(self, query: str) -> List[Document]:
        """Retrieves relevant document chunks for a query, using cache and reranking if enabled."""
        start_time = time.time()

        # Use the internal cached function, passing relevant config parameters as arguments
        # This ensures the cache key reflects the configuration used for retrieval/reranking.
        final_docs_with_scores = self._cached_retrieve_rerank(
            query,
            self.config.top_k, # Initial retrieval count
            self.config.use_hybrid_search,
            self.config.hybrid_alpha,
            self.config.use_reranking,
            self.config.rerank_top_n # Final count after reranking
        )

        # Extract only the Document objects from the final (Document, score) tuples
        final_docs = [doc for doc, score in final_docs_with_scores]

        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(final_docs)} documents for query in {retrieval_time:.3f} seconds.")
        logger.debug(f"Retrieved doc IDs: {[d.id for d in final_docs]}")

        return final_docs

    def retrieve_with_parent_context(self, query: str) -> List[Document]:
        """
        Retrieve using child chunks (precise) but return parent content (full context).
        
        This is the KEY to parent-child chunking:
        1. Search small child chunks (256 tokens) - gets precise matches
        2. Return large parent chunks (1000 tokens) - gives full context to LLM
        
        Returns:
            List of Documents with parent content (but child metadata/scores)
        """
        start_time = time.time()
        
        # Step 1: Retrieve child chunks normally (precise retrieval)
        child_docs = self.retrieve(query)
        
        if not child_docs:
            return []
        
        # Step 2: Expand children to parent content
        enriched_docs = []
        seen_parents = set()  # Avoid duplicate parents
        
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get('parent_id')
            
            # Skip if we already added this parent
            if parent_id and parent_id in seen_parents:
                logger.debug(f"Skipping duplicate parent: {parent_id}")
                continue
            
            if parent_id:
                seen_parents.add(parent_id)
            
            # Get parent content from map
            parent_content = self.vector_store.parent_map.get(child_doc.id)
            
            if parent_content:
                # Create new doc with parent content but child metadata/score
                enriched_doc = Document(
                    id=child_doc.id,  # Keep child ID for tracking
                    text=parent_content,  # ← FULL PARENT CONTEXT!
                    metadata={
                        **child_doc.metadata,
                        'expanded_to_parent': True,
                        'original_child_id': child_doc.id
                    }
                )
                enriched_doc.score = child_doc.score  # Preserve retrieval score
                enriched_docs.append(enriched_doc)
                logger.debug(f"Expanded child {child_doc.id} to parent (parent content: {len(parent_content)} chars)")
            else:
                # No parent mapping found, use child as-is
                logger.debug(f"No parent mapping for {child_doc.id}, using child content")
                enriched_docs.append(child_doc)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(child_docs)} children → expanded to {len(enriched_docs)} parents in {retrieval_time:.3f}s")
        
        return enriched_docs

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Answers a question using the full RAG pipeline: Retrieve -> Rerank -> Generate."""
        full_start_time = time.time()
        logger.info(f"Received query: '{query}'")

        # 1. Retrieve relevant documents (includes cache, hybrid search, reranking internally)
        retrieval_start_time = time.time()
        relevant_docs = self.retrieve(query) # Handles retrieval, reranking, caching
        retrieval_time = time.time() - retrieval_start_time

        if not relevant_docs:
            logger.warning("No relevant documents found after retrieval/reranking.")
            total_time = time.time() - full_start_time
            return {
                "query": query,
                "answer": "Could not find relevant information to answer the question.",
                "sources": [],
                "metrics": {
                    "retrieval_time_sec": retrieval_time,
                    "generation_time_sec": 0.0,
                    "total_time_sec": total_time,
                    "retrieved_doc_count_final": 0,
                }
            }

        # 2. Generate answer using the LLM (local or API)
        generation_start_time = time.time()
        llm_response = self.response_generator.generate_answer(query, relevant_docs)
        # llm_response contains 'answer', 'sources', 'metrics', '_debug_prompt'

        generation_time = llm_response.get("metrics", {}).get("generation_time_sec", time.time() - generation_start_time) # Use reported time if available

        total_time = time.time() - full_start_time

        # Combine results and metrics
        final_result = {
            "query": query,
            "answer": llm_response.get("answer", "Error: No answer generated."),
            "sources": llm_response.get("sources", []), # List of dicts {'id': ..., 'metadata': ...}
            "metrics": {
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": generation_time,
                "total_time_sec": total_time,
                "retrieved_doc_count_final": len(relevant_docs), # Count after all retrieval/reranking steps
                # Merge LLM-specific metrics (like model used, method)
                **(llm_response.get("metrics", {}))
            }
            # Optionally include debug prompt: "_debug_prompt": llm_response.get("_debug_prompt")
        }

        logger.info(f"Answer generated in {total_time:.3f} seconds. Final document count: {len(relevant_docs)}.")
        return final_result

    def retrieve_parallel(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieves documents using 5 parallel strategies with RRF fusion.
        
        This method leverages ParallelRetrievalFusion to run semantic, BM25, hybrid,
        expanded query, and entity-focused retrieval simultaneously, then fuses results
        using Reciprocal Rank Fusion for optimal diversity and relevance.
        
        Args:
            query: The search query
            top_k: Number of final documents to return (default: 5)
            
        Returns:
            List of top-k fused Document objects
        """
        logger.info(f"Using Parallel Retrieval Fusion for query: '{query}' (top_k={top_k})")
        try:
            # Run async parallel retrieval in sync context
            docs = asyncio.run(self.parallel_retriever.retrieve_parallel(query, top_k))
            logger.info(f"Parallel retrieval returned {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Parallel retrieval failed: {e}. Falling back to standard retrieval.")
            # Fallback to standard retrieval if parallel fails
            return self.retrieve(query)

    def answer_question_parallel(self, query: str) -> Dict[str, Any]:
        """
        Answers a question using Parallel Retrieval Fusion instead of standard retrieval.
        
        This method uses 5 parallel retrieval strategies (semantic, BM25, hybrid, 
        expanded query, entity-focused) fused with RRF, which should provide 
        better context quality and diversity, leading to improved answer quality.
        
        Args:
            query: The question to answer
            
        Returns:
            Dict containing:
                - query: Original query
                - answer: Generated answer
                - sources: List of source documents
                - metrics: Performance metrics including retrieval method used
        """
        full_start_time = time.time()
        logger.info(f"Received query (PARALLEL MODE): '{query}'")

        # 1. Retrieve using Parallel Fusion (with parent-child expansion if enabled)
        retrieval_start_time = time.time()
        relevant_docs = self.retrieve_parallel(query, top_k=self.config.top_k)
        
        # 1b. Expand to parent context if parent-child chunking enabled
        if self.config.use_parent_child_chunking and self.vector_store.parent_map:
            logger.info("Expanding child chunks to parent context...")
            enriched_docs = []
            seen_parents = set()
            
            for doc in relevant_docs:
                parent_id = doc.metadata.get('parent_id')
                if parent_id and parent_id in seen_parents:
                    continue
                if parent_id:
                    seen_parents.add(parent_id)
                
                parent_content = self.vector_store.parent_map.get(doc.id)
                if parent_content:
                    enriched_doc = Document(
                        id=doc.id,
                        text=parent_content,
                        metadata={**doc.metadata, 'expanded_to_parent': True}
                    )
                    enriched_doc.score = doc.score
                    enriched_docs.append(enriched_doc)
                else:
                    enriched_docs.append(doc)
            
            relevant_docs = enriched_docs
            logger.info(f"Expanded to {len(relevant_docs)} parent contexts")
        
        retrieval_time = time.time() - retrieval_start_time

        if not relevant_docs:
            logger.warning("No relevant documents found after parallel retrieval.")
            total_time = time.time() - full_start_time
            return {
                "query": query,
                "answer": "Could not find relevant information to answer the question.",
                "sources": [],
                "metrics": {
                    "retrieval_method": "parallel_fusion",
                    "retrieval_time_sec": retrieval_time,
                    "generation_time_sec": 0.0,
                    "total_time_sec": total_time,
                    "retrieved_doc_count_final": 0,
                }
            }

        # 2. Generate answer using the LLM
        generation_start_time = time.time()
        llm_response = self.response_generator.generate_answer(query, relevant_docs)
        generation_time = llm_response.get("metrics", {}).get("generation_time_sec", time.time() - generation_start_time)

        total_time = time.time() - full_start_time

        # Combine results and metrics
        final_result = {
            "query": query,
            "answer": llm_response.get("answer", "Error: No answer generated."),
            "sources": llm_response.get("sources", []),
            "metrics": {
                "retrieval_method": "parallel_fusion",  # Indicate parallel fusion was used
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": generation_time,
                "total_time_sec": total_time,
                "retrieved_doc_count_final": len(relevant_docs),
                **(llm_response.get("metrics", {}))
            }
        }

        logger.info(f"Answer generated (PARALLEL MODE) in {total_time:.3f} seconds. Final document count: {len(relevant_docs)}.")
        return final_result

    def save_settings(self, filepath: str = "rag_settings.json") -> None:
        """Saves the current RAG configuration to a JSON file."""
        # (Unchanged logic)
        logger.info(f"Saving RAG configuration to {filepath}")
        try:
            # Ensure directory exists
             os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
             # Convert dataclass to dict for JSON serialization
             config_dict = field_to_dict(self.config) # Use helper if complex types exist
             with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to save settings to {filepath}: {e}")
        except TypeError as e:
            logger.error(f"Failed to serialize config to JSON (potentially non-serializable types): {e}")

    @classmethod
    def load_from_settings(cls, filepath: str = "rag_settings.json") -> 'AdvancedRAG':
        """Loads RAG system configuration from a JSON file and initializes the system."""
        # (Unchanged logic)
        logger.info(f"Loading RAG configuration from {filepath}")
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)

            # Create RAGConfig instance from the loaded dict
            config = RAGConfig(**config_dict)
            # Initialize the RAG system with the loaded config
            return cls(config)
        except FileNotFoundError:
            logger.error(f"Settings file not found at {filepath}. Returning RAG with default settings.")
            return cls() # Return default initialized RAG
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON from {filepath}: {e}. Returning RAG with default settings.")
             return cls()
        except TypeError as e:
             logger.error(f"Error creating RAGConfig from settings (likely missing/incorrect keys or invalid types): {e}. Returning RAG with default settings.")
             return cls()
        except Exception as e:
             logger.error(f"An unexpected error occurred while loading settings: {e}. Returning RAG with default settings.", exc_info=True)
             return cls()

    def run_evaluation(self, eval_dataset_path: Optional[str] = None, sample_size: int = 100) -> Dict[str, Any]:
        """Runs evaluation on a dataset (expects columns: question, context, reference_answer)."""
        # (Unchanged logic, but ensure dependencies checked)
        if not DATASETS_AVAILABLE and not eval_dataset_path:
            logger.error("Evaluation requires the 'datasets' library or a specified dataset path.")
            return {"error": "Evaluation dependencies not met."}
        if not ROUGE_AVAILABLE:
             logger.warning("ROUGE library not found. Generation evaluation will be skipped/limited.")

        eval_data = None
        try:
            if eval_dataset_path:
                logger.info(f"Attempting to load evaluation data from: {eval_dataset_path}")
                if eval_dataset_path.endswith(".csv"):
                     eval_data = pd.read_csv(eval_dataset_path)
                elif eval_dataset_path.endswith(".jsonl") or eval_dataset_path.endswith(".json"):
                     eval_data = pd.read_json(eval_dataset_path, lines=eval_dataset_path.endswith(".jsonl"))
                elif eval_dataset_path.endswith(".parquet"):
                     eval_data = pd.read_parquet(eval_dataset_path)
                else:
                     raise ValueError(f"Unsupported file format: {eval_dataset_path}. Use CSV, JSON(L), or Parquet.")
                logger.info(f"Loaded evaluation data from {eval_dataset_path}. Rows: {len(eval_data)}")
            elif DATASETS_AVAILABLE:
                logger.info("Loading default SQuAD validation dataset for evaluation.")
                # Load fewer samples by default if using datasets library
                load_n = min(sample_size, 1000) # Limit dataset loading size
                dataset = load_dataset("squad", split=f"validation[:{load_n}]")
                eval_data = pd.DataFrame({
                    "id": dataset["id"],
                    "question": dataset["question"],
                    "context": dataset["context"],
                    "reference_answer": [ans['text'][0] if ans['text'] else "" for ans in dataset["answers"]]
                })
                logger.info(f"Loaded {len(eval_data)} samples from SQuAD validation set.")
            else:
                 raise RuntimeError("Cannot load evaluation data. Provide a path or install 'datasets'.")

            # --- Validate DataFrame ---
            required_cols = ["question", "context", "reference_answer"]
            missing_cols = [col for col in required_cols if col not in eval_data.columns]
            if missing_cols:
                 raise ValueError(f"Evaluation data is missing required columns: {missing_cols}")

            # Ensure correct types (optional but good practice)
            eval_data['question'] = eval_data['question'].astype(str)
            eval_data['context'] = eval_data['context'].astype(str)
            eval_data['reference_answer'] = eval_data['reference_answer'].astype(str)
            eval_data.dropna(subset=required_cols, inplace=True) # Drop rows with missing crucial info

            if len(eval_data) == 0:
                 raise ValueError("No valid evaluation data remaining after cleaning.")

            # --- Sample Data ---
            if len(eval_data) > sample_size:
                 logger.info(f"Sampling {sample_size} rows from the {len(eval_data)} available evaluation data points.")
                 eval_data = eval_data.sample(n=sample_size, random_state=42) # Use random state for reproducibility

        except Exception as e:
            logger.error(f"Failed to load or prepare evaluation data: {e}", exc_info=True)
            return {"error": f"Failed to load evaluation data: {e}"}

        # --- Evaluation Loop ---
        results = {
            "retrieval_precision": [], "retrieval_recall": [], "retrieval_f1": [],
            "generation_rouge1": [], "generation_rouge2": [], "generation_rougeL": [],
            "retrieval_time": [], "generation_time": [], "total_time": [],
            "errors": [], "processed_count": 0
        }
        # Store mapping from context_id to its generated chunk_ids for accurate retrieval eval
        context_to_chunk_ids: Dict[str, List[str]] = {}

        for i, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Evaluating RAG performance"):
            try:
                question = row['question']
                context = row['context']
                reference_answer = row['reference_answer']
                # Use a hash of the context as a stable ID for this evaluation item's source document
                context_id = f"eval_ctx_{hashlib.md5(context.encode()).hexdigest()}"

                # 1. Add context to store IF NOT ALREADY ADDED & Get relevant chunk IDs
                if context_id not in context_to_chunk_ids:
                    logger.debug(f"Adding new evaluation context {context_id} to vector store.")
                    # Add document, explicitly passing the context_id as the doc_id
                    chunk_ids = self.add_document(
                        context,
                        metadata={"source": "evaluation", "eval_set_id": row.get('id', f"index_{i}")},
                        doc_id=context_id
                    )
                    if not chunk_ids:
                         logger.warning(f"Failed to add/chunk evaluation context {context_id} for question '{question[:50]}...'. Skipping evaluation for this item.")
                         results["errors"].append({"row_index": i, "id": row.get('id'), "error": "Context chunking failed"})
                         continue # Skip this data point
                    context_to_chunk_ids[context_id] = chunk_ids
                    relevant_chunk_ids = chunk_ids
                else:
                    # Context already added, retrieve its known chunk IDs
                    relevant_chunk_ids = context_to_chunk_ids[context_id]
                    logger.debug(f"Using existing chunk IDs for context {context_id}")

                if not relevant_chunk_ids:
                     logger.warning(f"No relevant chunk IDs found for context {context_id}. Cannot evaluate retrieval accurately for this item.")
                     # Decide whether to proceed with generation eval only, or skip
                     # Let's proceed but mark retrieval metrics as NaN

                # 2. Perform RAG query
                rag_result = self.answer_question(question)
                generated_answer = rag_result['answer']
                # sources are [{'id': ..., 'metadata': ...}, ...]
                retrieved_sources = rag_result['sources']
                metrics = rag_result['metrics']

                # Create dummy Document objects just for the evaluator interface
                retrieved_docs_for_eval = [
                    Document(id=src['id'], text="", metadata=src.get('metadata', {}))
                    for src in retrieved_sources
                ]

                # 3. Evaluate Retrieval
                if relevant_chunk_ids: # Only evaluate if we know what *should* have been retrieved
                    retrieval_scores = self.evaluator.evaluate_retrieval(retrieved_docs_for_eval, relevant_chunk_ids)
                    results["retrieval_precision"].append(retrieval_scores["precision"])
                    results["retrieval_recall"].append(retrieval_scores["recall"])
                    results["retrieval_f1"].append(retrieval_scores["f1"])
                else:
                     results["retrieval_precision"].append(np.nan)
                     results["retrieval_recall"].append(np.nan)
                     results["retrieval_f1"].append(np.nan)


                # 4. Evaluate Generation (if ROUGE is available)
                if ROUGE_AVAILABLE:
                    generation_scores = self.evaluator.evaluate_generation(generated_answer, reference_answer)
                    results["generation_rouge1"].append(generation_scores.get("rouge-1", np.nan))
                    results["generation_rouge2"].append(generation_scores.get("rouge-2", np.nan))
                    results["generation_rougeL"].append(generation_scores.get("rouge-l", np.nan))
                    if "error" in generation_scores: # Log ROUGE specific errors
                         results["errors"].append({"row_index": i, "id": row.get('id'), "error": f"ROUGE error: {generation_scores['error']}"})
                else:
                    # Append NaN if ROUGE is not available
                    results["generation_rouge1"].append(np.nan)
                    results["generation_rouge2"].append(np.nan)
                    results["generation_rougeL"].append(np.nan)


                # 5. Record timings and count success
                results["retrieval_time"].append(metrics.get("retrieval_time_sec", np.nan))
                results["generation_time"].append(metrics.get("generation_time_sec", np.nan))
                results["total_time"].append(metrics.get("total_time_sec", np.nan))
                results["processed_count"] += 1

            except Exception as e:
                logger.error(f"Error during evaluation for row {i} (ID: {row.get('id')}): {e}", exc_info=True)
                results["errors"].append({"row_index": i, "id": row.get('id'), "error": str(e)})
                # Append NaNs to all metric lists to maintain structure
                for key in ["retrieval_precision", "retrieval_recall", "retrieval_f1",
                            "generation_rouge1", "generation_rouge2", "generation_rougeL",
                            "retrieval_time", "generation_time", "total_time"]:
                    results[key].append(np.nan)


        # --- Calculate Average Metrics ---
        avg_metrics = {}
        for key, values in results.items():
            if key not in ["errors", "processed_count"] and isinstance(values, list):
                # Use nanmean to ignore NaNs in averages
                valid_values = [v for v in values if v is not None and not np.isnan(v)]
                avg_metrics[f"avg_{key}"] = np.mean(valid_values) if valid_values else 0.0 # Use 0 if no valid values

        summary = {
            "config_used": field_to_dict(self.config), # Include config used for the eval run
            "evaluation_samples_requested": sample_size,
            "evaluation_samples_processed": results["processed_count"],
            "average_metrics": avg_metrics,
            "errors_count": len(results["errors"]),
            "errors_details": results["errors"][:20] # Limit detailed errors listed
        }

        logger.info(f"Evaluation completed. Processed: {results['processed_count']}/{len(eval_data)}. Errors: {len(results['errors'])}.")
        logger.info(f"Average Metrics: {json.dumps(avg_metrics, indent=2)}")
        if results["errors"]:
             logger.warning(f"Encountered {len(results['errors'])} errors during evaluation. First few: {results['errors'][:3]}")

        return summary

# Helper to convert dataclass to dict, handling potential complex types if needed
def field_to_dict(obj):
    # Basic implementation, enhance if complex types are stored in config
    if hasattr(obj, "__dict__"):
        return vars(obj)
    elif hasattr(obj, "__slots__"):
         return {slot: getattr(obj, slot) for slot in obj.__slots__}
    return obj # Fallback for non-dataclass/object types


# --- Example Usage ---

if __name__ == "__main__":
    logger.info("--- Starting Advanced RAG Example ---")

    # --- Configuration ---
    # <<< CHOOSE YOUR LLM MODE HERE >>>
    USE_INFERENCE_API = False # Set True for Llama 3 API, False for local Flan-T5

    # Ensure HUGGINGFACE_API_KEY is set in your environment (e.g., .env file)
    # load_dotenv() called at the top of the script

    config = None # Initialize config variable

    if USE_INFERENCE_API:
         logger.info("--- Configuring RAG for Hugging Face Inference API ---")
         if not HUGGINGFACE_HUB_AVAILABLE:
              raise ImportError("Cannot use Inference API: 'huggingface_hub' is not installed. Install it via 'pip install huggingface_hub'")
         hf_token = os.getenv("HUGGINGFACE_API_KEY")
         if not hf_token:
             raise ValueError("Cannot use Inference API: 'HUGGINGFACE_API_KEY' environment variable not set or empty.")
         logger.info(f"Found Hugging Face API token environment variable.")

         config = RAGConfig(
             # Embedding/reranker models usually remain local
             embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
             reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",

             # --- LLM Configuration for Inference API ---
             use_inference_client=True,
             inference_client_model="meta-llama/Meta-Llama-3-8B-Instruct", # Target Llama 3
             llm_model_name="meta-llama/Meta-Llama-3-8B-Instruct", # Also set base name for potential reference
             llm_max_new_tokens=300, # Allow more tokens for Llama 3
             llm_temperature=0.6,
             llm_top_p=0.9,
             hf_api_token_env_var="HUGGINGFACE_API_KEY", # Matches env var name

             # --- Other Settings ---
             chunk_strategy="paragraph_sentence",
             chunk_size=384,
             chunk_overlap=64,
             top_k=8,
             rerank_top_n=3,
             use_hybrid_search=True,
             hybrid_alpha=0.6,
             use_reranking=True,
             collection_name="rag_llama3_api_docs_v3", # Specific collection name
             persist_directory="./chroma_db_llama3_api_v3", # Specific persistence path
             use_gpu=torch.cuda.is_available() # GPU still relevant for local embedding/reranking
         )
    else:
         logger.info("--- Configuring RAG for Local Model (Flan-T5) ---")
         config = RAGConfig(
             embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
             reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
             llm_model_name="google/flan-t5-base", # Local model ID
             use_inference_client=False, # Explicitly false

             # Other settings can be the same or different
             chunk_strategy="paragraph_sentence",
             chunk_size=384,
             chunk_overlap=64,
             top_k=8,
             rerank_top_n=3,
             use_hybrid_search=True,
             hybrid_alpha=0.6,
             use_reranking=True,
             collection_name="rag_flant5_local_docs_v3",
             persist_directory="./chroma_db_flant5_local_v3",
             use_gpu=torch.cuda.is_available(), # Use GPU if available for local model
             llm_max_new_tokens=150 # Flan-T5 might need fewer tokens
         )

    # Save the final chosen config (optional)
    config_save_path = f"rag_settings_{'api' if USE_INFERENCE_API else 'local'}.json"
    logger.info(f"Using configuration: {config}")
    # AdvancedRAG will save it if save_settings is called

    # --- Initialize RAG System ---
    try:
        rag = AdvancedRAG(config)
    except Exception as init_error:
         logger.error(f"FATAL: Failed to initialize AdvancedRAG system: {init_error}", exc_info=True)
         exit(1) # Exit if core components fail to initialize


    # --- Add Documents ---
    logger.info("\n--- Adding Documents ---")
    try:
        # Use get_collection to check existence without triggering load_corpus
        collection_exists = False
        try:
            existing_collection = rag.vector_store.client.get_collection(rag.config.collection_name)
            initial_doc_count = existing_collection.count()
            collection_exists = True
            logger.info(f"Existing collection '{rag.config.collection_name}' found with {initial_doc_count} documents.")
        except Exception: # Catches errors if collection doesn't exist
             initial_doc_count = 0
             logger.info(f"Collection '{rag.config.collection_name}' not found or error accessing. Will create/add documents.")

        # Add documents only if the collection is empty or very small
        if initial_doc_count < 5: # Arbitrary threshold
            logger.info("Adding new documents to the collection...")
            doc1_chunks = rag.add_document(
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
                "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
                "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.",
                {"title": "Eiffel Tower Facts", "category": "landmarks", "year": 1889},
                doc_id="eiffel_tower_doc" # Provide a specific ID
            )
            logger.info(f"Added Document 1 (Eiffel Tower) chunks: {len(doc1_chunks)}")

            doc2_chunks = rag.add_document(
                "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. "
                "Machine learning (ML) is a subset of AI that focuses on the development of systems that can learn from and make decisions based on data. Deep learning is a further subset of ML that uses neural networks with many layers (deep neural networks) to analyze various factors of data.",
                {"title": "AI Definitions", "category": "technology", "concepts": ["AI", "ML", "Deep Learning"]},
                doc_id="ai_definitions_doc"
            )
            logger.info(f"Added Document 2 (AI Definitions) chunks: {len(doc2_chunks)}")

            doc3_chunks = rag.add_document(
                "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organisms' activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.",
                {"title": "Photosynthesis Explained", "category": "biology"},
                doc_id="photosynthesis_doc"
            )
            logger.info(f"Added Document 3 (Photosynthesis) chunks: {len(doc3_chunks)}")

            # Example using DataFrame
            data = {
                'id': ['bio001', 'chem001'],
                'article_text': [
                    "Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert chemical energy from oxygen molecules or nutrients into adenosine triphosphate (ATP), and then release waste products.",
                    "The periodic table is a tabular arrangement of the chemical elements, ordered by their atomic number, electron configuration, and recurring chemical properties. Elements are arranged in order of increasing atomic number."
                ],
                'topic': ['Biology', 'Chemistry'],
                'keywords': [['metabolism', 'ATP', 'cells'], ['elements', 'atomic number', 'chemistry']]
            }
            df = pd.DataFrame(data)
            added_df_chunks, failed_df_docs = rag.add_documents_from_dataframe(
                df,
                text_column='article_text',
                metadata_columns=['topic', 'keywords'],
                id_column='id'
            )
            logger.info(f"Added {len(added_df_chunks)} DataFrame chunks.")
            if failed_df_docs: logger.warning(f"Failed DataFrame docs: {failed_df_docs}")
        else:
            logger.info("Skipping document addition as collection seems populated.")
            # Ensure BM25 is loaded if collection already existed and hybrid search is on
            if rag.config.use_hybrid_search and rag.vector_store._bm25 is None:
                 logger.info("Attempting to load corpus for BM25 for existing collection...")
                 rag.vector_store._load_corpus_for_bm25()


    except Exception as doc_add_err:
         logger.error(f"Error during document loading/addition phase: {doc_add_err}", exc_info=True)
         # Continue to querying if possible, but warn user
         logger.warning("Proceeding to querying, but document store might be incomplete.")


    # --- Querying ---
    logger.info("\n--- Answering Questions ---")

    queries = [
        "Who designed the Eiffel Tower?",
        "What is the relationship between AI, ML, and Deep Learning?",
        "How do plants store energy?",
        "Tell me about the periodic table.",
        "What is the process that creates ATP?", # Answer is in cellular respiration text
        "What is the capital of Spain?" # Should not be answerable from context
    ]

    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1} ---")
        logger.info(f"Processing Query: {query}")
        try:
            result = rag.answer_question(query)
            print(f"Query: {query}")
            print(f"Answer: {result['answer']}")
            print(f"Sources Used ({len(result['sources'])}):")
            for src in result['sources']:
                 print(f"  - ID: {src['id']} (Title: {src.get('metadata', {}).get('title', 'N/A')})")
            print(f"Metrics: {result['metrics']}")
            # print(f"DEBUG Prompt: {result.get('_debug_prompt', 'N/A')}") # Uncomment to see prompt
        except Exception as query_err:
             logger.error(f"Failed to answer query '{query}': {query_err}", exc_info=True)
             print(f"Query: {query}")
             print(f"Answer: Error processing this query. See logs.")
        print("-" * 15)
        time.sleep(1) # Add a small delay between queries, especially if using API


    # --- Optional: Evaluation ---
    logger.info("\n--- Running Evaluation ---")
    # Create a dummy evaluation CSV file if it doesn't exist
    eval_file = "rag_eval_data.csv"
    if not os.path.exists(eval_file):
        eval_df = pd.DataFrame({
            'id': ['eval_01', 'eval_02', 'eval_03', 'eval_04'],
            'question': [
                "When was the Eiffel Tower constructed?",
                "What process converts light energy into chemical energy in plants?",
                "What is ATP related to?",
                "How are chemical elements ordered in the periodic table?"
            ],
            'context': [ # Provide the context where the answer should be found
                 "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair...",
                 "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organisms' activities.",
                 "Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert chemical energy from oxygen molecules or nutrients into adenosine triphosphate (ATP), and then release waste products.",
                 "The periodic table is a tabular arrangement of the chemical elements, ordered by their atomic number, electron configuration, and recurring chemical properties. Elements are arranged in order of increasing atomic number."
            ],
            'reference_answer': [
                "1887 to 1889",
                "Photosynthesis",
                "Cellular respiration",
                "By their atomic number"
            ]
        })
        try:
             eval_df.to_csv(eval_file, index=False)
             logger.info(f"Created dummy evaluation file: {eval_file}")
        except Exception as e:
             logger.error(f"Failed to create dummy evaluation file {eval_file}: {e}")
             eval_file = None # Prevent trying to load it later if creation failed

    # Check if dependencies and file are available before running
    if ROUGE_AVAILABLE and eval_file and os.path.exists(eval_file):
        try:
            evaluation_results = rag.run_evaluation(eval_dataset_path=eval_file, sample_size=10) # Evaluate on small sample
            print("\n--- Evaluation Summary ---")
            # Use json dumps for pretty printing the potentially nested dict
            print(json.dumps(evaluation_results, indent=2, default=str)) # Use default=str for non-serializable items like numpy floats

            # Save evaluation results
            eval_results_path = f"rag_evaluation_results_{'api' if USE_INFERENCE_API else 'local'}.json"
            try:
                 with open(eval_results_path, 'w') as f:
                      json.dump(evaluation_results, f, indent=2, default=str)
                 logger.info(f"Evaluation results saved to {eval_results_path}")
            except Exception as save_err:
                 logger.error(f"Failed to save evaluation results: {save_err}")

        except Exception as eval_err:
            logger.error(f"Evaluation run failed: {eval_err}", exc_info=True)
            print("\n--- Evaluation Failed ---")
            print(f"Error: {eval_err}")
    else:
        logger.warning("Skipping evaluation due to missing dependencies (rouge), missing eval file, or file creation failure.")

    # --- Save Final Config ---
    try:
         rag.save_settings(config_save_path)
         logger.info(f"Final configuration saved to {config_save_path}")
    except Exception as save_err:
         logger.error(f"Failed to save final configuration: {save_err}")

    logger.info("\n--- Advanced RAG Example Finished ---")