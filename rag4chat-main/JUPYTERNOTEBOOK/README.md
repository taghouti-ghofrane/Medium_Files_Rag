# RAGAnything Test and Configuration Notebook

This folder contains a Jupyter notebook for testing and configuring RAGAnything step by step.

## File

- `RAGAnything_Test_and_Configuration.ipynb` - Complete notebook with examples

## Prerequisites

1. **Install Jupyter Notebook or JupyterLab:**
   ```bash
   pip install jupyter
   # or
   pip install jupyterlab
   ```

2. **Install required packages:**
   ```bash
   pip install raganything lightrag sentence-transformers openai numpy aiohttp python-dotenv
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

## How to Use

1. **Start Jupyter:**
   ```bash
   cd JUPYTERNOTEBOOK
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open the notebook:**
   - Open `RAGAnything_Test_and_Configuration.ipynb`

3. **Run cells step by step:**
   - Execute cells in order from top to bottom
   - Each section builds on the previous one
   - Read the markdown cells for explanations

## Notebook Sections

### 1. Setup and Installation
- Check Python version
- Verify required packages are installed

### 2. Configuration
- Load configuration from `config/settings.py`
- Display current settings
- Verify API keys

### 3. Initialize RAGAnything
- Define helper functions (embedding, LLM, vision)
- Create RAGAnything configuration
- Initialize RAGAnything instance

### 4. Add Documents
- Add single documents (PDF, DOCX, TXT, MD)
- Add multiple documents in batch
- Handle file paths and errors

### 5. Query Documents
- Simple queries
- Queries with custom parameters
- Batch queries

### 6. Advanced Configuration
- Update RAGAnything configuration
- Configure LightRAG parameters
- Use different embedding models

### 7. Visualization
- Inspect storage directory
- Test embedding function
- Test LLM function

## Example Usage

### Basic Example

```python
# After running initialization cells:

# Add a document
result = await rag.ainsert("../example/example.pdf")
print(result)

# Query
response = await rag.aquery("What is the main topic?")
print(response)
```

### Advanced Example

```python
# Create custom configuration
config = RAGAnythingConfig(
    working_dir="./my_storage",
    parser="mineru",
    enable_image_processing=True,
)

# Initialize with custom LightRAG parameters
rag = RAGAnything(
    config=config,
    llm_model_func=llm_model_func,
    vision_model_func=vision_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs={
        "top_k": 10,
        "cosine_threshold": 0.6,
    }
)
```

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Make sure you're running from the notebook directory
   - Check that project root is in Python path (first cell handles this)

2. **API Key Errors:**
   - Verify `.env` file exists in project root
   - Check that `OPENAI_API_KEY` is set correctly
   - Ensure API key starts with `sk-`

3. **Document Processing Errors:**
   - Check that MinerU is installed for PDF processing
   - Verify file paths are correct
   - Check logs in storage directory

4. **Embedding Model Errors:**
   - First run will download the model (may take time)
   - Ensure internet connection is available
   - Check available disk space

### Getting Help

- Check the logs in `logs/` directory
- Review error messages in notebook output
- Verify configuration in `config/settings.py`
- Check RAGAnything documentation in `src/raganything/README.md`

## Notes

- The notebook uses async/await syntax - make sure cells are run in order
- Storage directories are created automatically
- First document processing may take longer (model downloads, initialization)
- Embedding models are cached after first load

## Next Steps

After completing the notebook:

1. Experiment with different embedding models
2. Adjust LightRAG parameters for your use case
3. Add your own documents
4. Test with different query types
5. Integrate into your application using `services/build_database.py` as reference

