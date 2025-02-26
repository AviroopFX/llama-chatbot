# Getting Started with Your Custom Llama Chatbot

This guide will help you set up and run your custom Llama chatbot that can be trained on your own data and answer questions based on information stored in SQLite.

## Prerequisites

1. Python 3.8 or higher
2. Sufficient disk space for the Llama model (typically 4-7GB depending on the model)

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download a Llama Model

The chatbot requires a Llama model in GGUF format. You can download one using the provided script:

```bash
python download_model.py
```

This will download the Llama-2-7B-Chat model (Q4_K_M quantization) which offers a good balance between performance and resource usage. If you want to use a different model, you can specify it:

```bash
python download_model.py --model llama-2-13b-chat.Q4_K_M.gguf --url https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf
```

### 3. Run the Application

You can run the application using the provided start script:

```bash
./start_sqlite.sh
```

Or you can run it directly:

```bash
python app_sqlite.py
```

The application will be available at http://localhost:8000

## Using the Chatbot

1. **Load a Model**: Select a model from the dropdown and click "Load Model"
2. **Upload Documents**: Use the upload form to add documents (PDF, DOCX, TXT)
3. **Ask Questions**: Type your questions in the chat interface
4. **View Sources**: The chatbot will show the sources it used to generate the response

## Customizing the Chatbot

### Using Different Models

You can use any Llama model in GGUF format. Smaller models will be faster but may provide less accurate responses. Larger models will be more accurate but require more resources.

### Adjusting Embedding Settings

If you want to adjust how documents are processed and embedded, you can modify the parameters in `data_processor.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `embedding_model`: Model used for generating embeddings (default: "all-MiniLM-L6-v2")

### Adjusting Llama Model Settings

You can adjust the Llama model parameters in `llama_model.py`:

- `context_size`: Maximum context size (default: 2048)
- `max_tokens`: Maximum number of tokens to generate (default: 512)
- `temperature`: Temperature for sampling (default: 0.7)
- `top_p`: Top-p sampling parameter (default: 0.95)

## Troubleshooting

### Model Loading Issues

If you have issues loading the model:
- Make sure the model file exists in the `models` directory
- Check that you have enough RAM (at least 8GB recommended)
- For larger models, you may need to adjust the `n_gpu_layers` parameter in `llama_model.py`

### Document Processing Issues

If you have issues processing documents:
- Check that the file format is supported (PDF, DOCX, TXT)
- Make sure the file is not corrupted
- For large files, you may need to increase the chunk size in `data_processor.py`
