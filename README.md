# Custom Llama Chatbot

A chatbot built using Llama that can be trained on custom data and answers questions based on information stored in SQLite.

## Features

- Train the chatbot on custom documents (PDF, DOCX, TXT, CSV, Excel, JSON)
- Store processed information in SQLite
- Generate responses using Llama
- Simple web interface for interaction

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download a Llama model (GGUF format) and place it in the `models` directory.
   You can download models from [TheBloke's Hugging Face page](https://huggingface.co/TheBloke).
   Recommended models:
   - [Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
   - [Llama-2-13B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF)

3. Start the application:
   ```
   python app_sqlite.py
   ```
   Or use the start script:
   ```
   ./start_sqlite.sh
   ```

4. Access the web interface at http://localhost:8000

## Supported File Formats

The chatbot can process and extract information from the following file formats:

- **PDF** (.pdf) - Extracts text from all pages
- **Word Documents** (.docx) - Extracts text content
- **Text Files** (.txt) - Processes plain text
- **CSV Files** (.csv) - Processes tabular data
- **Excel Files** (.xlsx, .xls) - Processes spreadsheets from all sheets
- **JSON Files** (.json) - Processes structured data

## Usage

1. Upload your documents through the web interface
2. The system will process and store the information in SQLite
3. Ask questions through the chat interface
4. The system will retrieve relevant information from the database and generate responses using Llama

## Auto-start on System Boot

To make the chatbot start automatically when your Mac starts:

1. The launch agent configuration has been set up at `~/Library/LaunchAgents/com.chatbot.app.plist`
2. The application logs can be found in:
   - Main log: `logs/app.log`
   - Error log: `logs/error.log`

To manually control the service:
- Stop: `launchctl unload ~/Library/LaunchAgents/com.chatbot.app.plist`
- Start: `launchctl load -w ~/Library/LaunchAgents/com.chatbot.app.plist`

## Project Structure

- `app_sqlite.py`: Main application file
- `data_processor.py`: Handles document processing and embedding
- `db_manager_sqlite.py`: Manages interactions with SQLite
- `llama_model.py`: Handles interactions with the Llama model
- `templates/`: Contains HTML templates for the web interface
- `static/`: Contains static files (CSS, JS) for the web interface
- `models/`: Directory to store Llama model files
- `data/`: Directory to store SQLite database and uploaded documents
