# Embedding System Migration - Open Source

## ✅ Migration Completed

The embedding system has been migrated from Alibaba Cloud Bailian API to **sentence-transformers** (open source).

## Changes Made

### 1. **services/build_database.py**
- Replaced `bailian_embed_async()` with `sentence_transformers_embed_async()`
- Updated `AsyncEmbeddingWrapper` to use sentence-transformers
- Removed dependency on `DASHSCOPE_API_KEY`

### 2. **config/settings.py**
- Changed default model: `BAAI/bge-large-en-v1.5` (1024 dimensions)
- Updated available models list with open source options

### 3. **services/get_top_from_rag.py**
- Replaced `bailian_embed_async()` with `sentence_transformers_embed_async()`
- Removed dependency on Alibaba API

## Installation

Install the required package:

```bash
pip install sentence-transformers
```

## Available Models

The system now supports these open source models (no API key needed):

1. **BAAI/bge-large-en-v1.5** (1024 dim) - **Default** - Best quality
2. **BAAI/bge-base-en-v1.5** (768 dim) - Good balance
3. **all-mpnet-base-v2** (768 dim) - General purpose
4. **all-MiniLM-L6-v2** (384 dim) - Fast, lightweight

## Benefits

✅ **No API key required** - Completely free and open source
✅ **Runs locally** - No external API calls
✅ **Fast batch processing** - Efficient embedding generation
✅ **High quality** - BGE models are state-of-the-art for RAG

## Notes

- Models are automatically downloaded from Hugging Face on first use
- Models are cached in memory to avoid reloading
- Embeddings are normalized for cosine similarity
- Dimension is automatically detected based on model

## Migration Complete

You can now use the system without `DASHSCOPE_API_KEY`!

