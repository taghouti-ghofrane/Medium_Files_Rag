import asyncio
import json
import os
import numpy as np
import base64
import io
from typing import List, Dict, Any, Tuple
# Removed AsyncOpenAI import - no longer needed for open source embeddings

# =======================
# Helper: Sentence Transformers Embedding (Open Source)
# =======================
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")

# Global model cache
_query_embedding_model_cache = {}

async def sentence_transformers_embed_async(text: str, model: str = "BAAI/bge-large-en-v1.5") -> List[float]:
    """
    Generate embedding using sentence-transformers (open source, no API key needed)
    
    Args:
        text: Text string to embed
        model: Model name from Hugging Face
    
    Returns:
        Embedding vector as list of floats
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is not installed. Install with: pip install sentence-transformers")
    
    # Load model (cache it)
    if model not in _query_embedding_model_cache:
        print(f"📥 Loading sentence-transformers model: {model}")
        _query_embedding_model_cache[model] = SentenceTransformer(model)
        print(f"✅ Model loaded successfully")
    
    model_instance = _query_embedding_model_cache[model]
    target_dim = model_instance.get_sentence_embedding_dimension()
    
    try:
        # Run in executor to make it async-friendly
        import asyncio
        loop = asyncio.get_event_loop()
        
        embedding = await loop.run_in_executor(
            None,
            lambda: model_instance.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )
        
        return embedding.tolist()
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        # Return zero vector to avoid interruption
        return [0.0] * target_dim

# =======================
# Helper: decode Base64 string to NumPy array
# =======================
# =======================
# Helper: decode Base64 string to NumPy array (fixed - supports pickle)
# =======================
# =======================
# Helper: decode Base64 string to NumPy array (fixed - direct float32)
# =======================
def decode_base64_vector_matrix(base64_str: str, num_vectors: int, vector_dim: int = 1024, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Decode Base64-encoded raw float32 vector data into a NumPy array.

    Assumes the raw data is contiguous binary for num_vectors vectors of length vector_dim with dtype.
    """
    try:
        # 1. Base64 decode
        decoded_bytes = base64.b64decode(base64_str)
        print(f"✅ Decoded matrix string into {len(decoded_bytes)} bytes.")

        # 2. Validate byte length
        expected_bytes = num_vectors * vector_dim * dtype().itemsize # itemsize for float32 is 4
        if len(decoded_bytes) != expected_bytes:
            print(f"⚠️ Warning: decoded bytes ({len(decoded_bytes)}) != expected ({expected_bytes}).")

        # 3. Re-interpret bytes as NumPy array (assumes little-endian)
        array_flat = np.frombuffer(decoded_bytes, dtype=dtype)
        
        # 4. Reshape to (num_vectors, vector_dim)
        array_matrix = array_flat.reshape((num_vectors, vector_dim))
        
        print(f"✅ Reshaped to NumPy array, shape {array_matrix.shape}, dtype {array_matrix.dtype}.")
        return array_matrix

    except Exception as e:
        print(f"❌ Error decoding/reshaping matrix: {e}")
        import traceback
        traceback.print_exc()
        raise  # re-raise for caller

# =======================
# Cosine similarity
# =======================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two NumPy vectors"""
    # Ensure 1D
    a = a.ravel()
    b = b.ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0 # avoid division by zero
    return np.dot(a, b) / (norm_a * norm_b)

# =======================
# Load vdb_chunks.json data
# =======================
def load_vdb_chunks(file_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Load vdb_chunks.json, decode Base64 matrix field, return (data list, decoded NumPy matrix).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks_data = data.get("data", [])
        raw_matrix_string = data.get("matrix", "")  # single string
        
        print(f"✅ Loaded {len(chunks_data)} chunks from data list.")

        if not raw_matrix_string:
             print("⚠️ Warning: 'matrix' field empty or missing.")
             return chunks_data, np.array([])

        # Decode Base64-encoded NumPy matrix
        try:
            num_chunks = len(chunks_data)
            matrix_data_np: np.ndarray = decode_base64_vector_matrix(raw_matrix_string, num_vectors=num_chunks, vector_dim=1024, dtype=np.float32)
        except Exception as e:
             print(f"❌ Failed to decode 'matrix' field: {e}")
             return chunks_data, np.array([])

        # Validate matrix shape
        if matrix_data_np.size > 0:
            expected_rows = len(chunks_data)
            actual_rows, actual_cols = matrix_data_np.shape
            if actual_rows != expected_rows:
                print(f"⚠️ Warning: matrix rows ({actual_rows}) != data length ({expected_rows}).")
            if actual_cols != 1024: # adjust if embedding dim changes
                 print(f"⚠️ Warning: matrix cols ({actual_cols}) not 1024.")
            else:
                 print(f"✅ Matrix shape OK: {matrix_data_np.shape} for {expected_rows} chunks.")
        else:
             print("⚠️ Warning: decoded matrix is empty.")

        return chunks_data, matrix_data_np

    except FileNotFoundError:
        print(f"❌ Error: file not found {file_path}")
        return [], np.array([])
    except json.JSONDecodeError as e:
        print(f"❌ Error: failed to parse JSON {file_path}: {e}")
        return [], np.array([])
    except Exception as e:
        print(f"❌ Unknown error while loading data: {e}")
        import traceback
        traceback.print_exc()
        return [], np.array([])

# =======================
# Main query and similarity computation
# =======================
async def query_and_find_topk(query_text: str, vdb_file_path: str = "./rag_storage_new/vdb_chunks.json", topk: int = 5):
    """
    Query text and find top-k most similar entries from vdb_chunks.json using pre-decoded vectors.
    """
    print(f"🔍 Loading data from: {vdb_file_path} ...")
    
    # 1. Load data and decoded vectors
    chunks_data, matrix_data_np = load_vdb_chunks(vdb_file_path)
    
    if not chunks_data or matrix_data_np.size == 0:
        print("⚠️ Warning: no data or vectors loaded.")
        return

    num_chunks = len(chunks_data)
    num_vectors = matrix_data_np.shape[0] if matrix_data_np.size > 0 else 0
    print(f"✅ Loaded {num_chunks} entries and vector matrix ({num_vectors} rows).")

    # 2. Get embedding for query text (sentence-transformers - open source)
    print("🔍 Generating query embedding (sentence-transformers)...")
    try:
        # Use the same model as configured in settings
        from config.settings import EMBEDDING_MODEL
        query_embedding_list = await sentence_transformers_embed_async(query_text, model=EMBEDDING_MODEL)
        # Convert to NumPy and ensure 1D shape
        query_embedding_np = np.array(query_embedding_list, dtype=np.float32).ravel()
        print(f"✅ Query embedding shape: {query_embedding_np.shape}")
    except ImportError as e:
        print(f"❌ Embedding library error: {e}")
        print("💡 Install with: pip install sentence-transformers")
        return
    except Exception as e:
        print(f"❌ Error getting query embedding: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Compute similarities
    print("🔍 Computing similarities...")
    similarities_and_chunks: List[Tuple[float, Dict[str, Any]]] = []

    # Loop for clarity; matrix operations could be faster. matrix_data_np shape (N, 1024)
    for i in range(matrix_data_np.shape[0]):
        try:
            # Embedding for chunk i
            content_embedding_np = matrix_data_np[i] # shape (1024,)
            # Cosine similarity
            sim = cosine_similarity(query_embedding_np, content_embedding_np)
            # Store similarity and chunk
            if i < len(chunks_data):
                 similarities_and_chunks.append((sim, chunks_data[i]))
            else:
                 print(f"❌ Index mismatch: matrix row {i} has no corresponding chunk.")
        except Exception as e:
            print(f"❌ Error processing index {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not similarities_and_chunks:
        print("⚠️ Warning: no similarities computed.")
        return

    # 4. Sort and get Top-K
    print(f"🔍 Sorting and taking Top-{topk}...")
    similarities_and_chunks.sort(key=lambda x: x[0], reverse=True)
    topk_results = similarities_and_chunks[:topk]

    # 5. Output results
    print("\n--- 📋 Query Results (Top-K) ---")
    print(f"🔍 Query text: {query_text}\n")
    # print(topk_results)
    for i, (similarity, chunk) in enumerate(topk_results):
        content = chunk.get("content", "N/A")
        file_path = chunk.get("file_path", "N/A")
        chunk_id = chunk.get("__id__", "N/A")
        created_at = chunk.get("__created_at__", "N/A")
        print(f"--- 🏆 Top {i+1} (similarity: {similarity:.4f}) ---")
        print(f"🆔 ID: {chunk_id}")
        print(f"🕒 Created at: {created_at}")
        print(f"📁 File path: {file_path}")
        # Preview first N characters
        print(f"📄 Content preview: {content[:500]}...\n")

    formatted_results = []
    for similarity, chunk in topk_results:
            result_item = {
                "similarity": similarity,
                "content": chunk.get("content", ""),
                "file_path": chunk.get("file_path", ""),
                "chunk_id": chunk.get("__id__", ""),
                # add more fields if needed
            }
            formatted_results.append(result_item)
    
    print("\n--- Query Results (Top-K) ---")
    print(f"Query text: {query_text}\n")
    for i, item in enumerate(formatted_results): # print using formatted_results
        content = item.get("content", "N/A")
        file_path = item.get("file_path", "N/A")
        chunk_id = item.get("chunk_id", "N/A")
        similarity_score = item.get("similarity", 0.0)
        print(f"--- Top {i+1} (similarity: {similarity_score:.4f}) ---")
        print(f"ID: {chunk_id}")
        print(f"File path: {file_path}")
        print(f"Content: {content[:500]}...\n")
        
    # Return formatted results
    return formatted_results
    

# =======================
# Entry point
# =======================
async def main():
    # --- Config ---
    query_text = "Semantic images?"
    vdb_file_path = "./rag_storage_new/vdb_chunks.json" # ensure path is correct
    topk = 5
    # --- End config ---

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ Error: environment variable DASHSCOPE_API_KEY not set. Please set and retry.")
        return # graceful exit

    result = await query_and_find_topk(query_text, vdb_file_path, topk)
    return result

# =======================
# Run
# =======================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user.")
    except Exception as e:
        print(f"❌ Unhandled error during execution: {e}")
        import traceback
        traceback.print_exc()
