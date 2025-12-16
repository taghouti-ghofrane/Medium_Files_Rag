# RAGAnything Initialization Guide

This guide explains Section 3: "Initialize RAGAnything" in detail.

## Overview

Section 3 consists of 4 main cells that set up and initialize RAGAnything:

1. **Cell 3**: Define helper functions (embedding, LLM, vision)
2. **Cell 4**: Create RAGAnything configuration
3. **Cell 5**: Create embedding function wrapper
4. **Cell 6**: Initialize RAGAnything instance
5. **Cell 7**: Verify initialization (NEW - added for testing)

## Step-by-Step Explanation

### Cell 3: Helper Functions

This cell defines three critical async functions:

#### 1. `sentence_transformers_embed_async()`
- **Purpose**: Converts text to embedding vectors using sentence-transformers
- **Model**: Uses `EMBEDDING_MODEL` from config (default: `BAAI/bge-large-en-v1.5`)
- **Features**:
  - Model caching (loads once, reuses)
  - Async execution (runs in executor thread)
  - Normalized embeddings (for cosine similarity)
- **Returns**: List of embedding vectors (each is a list of floats)

#### 2. `llm_model_func()`
- **Purpose**: Calls OpenAI API for text generation
- **Model**: Uses `DOC_PROCESSING_LLM_MODEL` from config (default: `gpt-4o-mini`)
- **Features**:
  - Supports system prompts
  - Supports conversation history
  - Filters unsupported kwargs (`hashing_kv` from lightrag)
- **Returns**: Generated text response

#### 3. `vision_model_func()`
- **Purpose**: Processes images with vision models
- **Model**: Uses `gpt-4o` for vision tasks
- **Features**:
  - Handles base64 encoded images
  - Falls back to LLM if no image provided
- **Returns**: Generated text response describing the image

### Cell 4: Configuration

Creates a `RAGAnythingConfig` object with:

```python
config = RAGAnythingConfig(
    working_dir="./rag_storage_test",  # Where to store RAG data
    parser="mineru",                    # Document parser (MinerU for PDFs)
    parse_method="auto",                 # Auto-detect document structure
    enable_image_processing=True,        # Process images in documents
    enable_table_processing=True,        # Extract tables
    enable_equation_processing=True,     # Extract equations
)
```

### Cell 5: Embedding Function

Wraps the embedding function in `EmbeddingFunc` format required by LightRAG:

```python
embedding_func = EmbeddingFunc(
    embedding_dim=1024,      # Dimension (1024 for bge-large, 768 for bge-base)
    max_token_size=512,      # Maximum tokens per text
    func=async_embedding_func,  # The async function
)
```

### Cell 6: Initialize RAGAnything

Creates the main RAGAnything instance:

```python
rag = RAGAnything(
    config=config,                    # Configuration object
    llm_model_func=llm_model_func,    # LLM function
    vision_model_func=vision_model_func,  # Vision function
    embedding_func=embedding_func,    # Embedding function
)
```

### Cell 7: Verification (NEW)

Tests that all components work correctly:
- Tests embedding function
- Tests LLM function
- Verifies RAGAnything instance

## Important Notes

### Dependencies

Make sure you've run:
1. **Cell 1**: Setup (adds project to path)
2. **Cell 2**: Configuration (loads settings)

### Error Handling

If initialization fails, check:
1. ✅ All packages installed (`sentence-transformers`, `openai`, etc.)
2. ✅ OpenAI API key set in `.env` file
3. ✅ Internet connection (for model downloads)
4. ✅ MinerU installed (for PDF parsing)

### Model Downloads

On first run:
- **Embedding model** will be downloaded (can take 5-10 minutes)
- Model is cached locally for future use
- Requires ~500MB-1GB disk space

### Storage Directory

- Created automatically at `./rag_storage_test`
- Contains:
  - Vector databases (`vdb_*.json`)
  - Key-value stores (`kv_store_*.json`)
  - Graph data (`graph_*.graphml`)
  - Document status (`kv_store_doc_status.json`)

## Troubleshooting

### Error: "EMBEDDING_MODEL not defined"
**Solution**: Run Cell 2 (Configuration) first

### Error: "OPENAI_API_KEY not defined"
**Solution**: 
1. Create `.env` file in project root
2. Add: `OPENAI_API_KEY=your-key-here`

### Error: "Model not found"
**Solution**: 
- Check internet connection
- Wait for model download to complete
- Check disk space

### Error: "MinerU not installed"
**Solution**: 
```bash
pip install mineru
# or
uv pip install mineru
```

## Next Steps

After successful initialization:
1. ✅ Proceed to Section 4: Add Documents
2. ✅ Test with a sample document
3. ✅ Query the documents in Section 5

## Advanced Configuration

You can customize initialization by:

1. **Changing embedding model**:
   ```python
   # In Cell 3, change:
   model = "BAAI/bge-base-en-v1.5"  # Faster, smaller
   ```

2. **Changing LLM model**:
   ```python
   # In config/settings.py:
   DOC_PROCESSING_LLM_MODEL = "gpt-4o"  # More powerful
   ```

3. **Custom LightRAG parameters**:
   ```python
   rag = RAGAnything(
       ...,
       lightrag_kwargs={
           "top_k": 10,
           "cosine_threshold": 0.6,
       }
   )
   ```

## Summary

Section 3 sets up:
- ✅ Embedding pipeline (sentence-transformers)
- ✅ LLM pipeline (OpenAI)
- ✅ Vision pipeline (OpenAI vision)
- ✅ RAGAnything instance
- ✅ Storage directory

Once complete, you're ready to add documents and query them!

