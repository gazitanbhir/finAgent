# AI Finance Agent with Voice, File Upload, and Tool Use

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Choose an appropriate license -->

**Live Demo:** [https://finagent-vc73.onrender.com/](https://finagent-vc73.onrender.com/)

## Overview

This project implements an AI-powered assistant designed to help with finance-related tasks. It features a web-based chat interface where users can interact using text, voice commands, or by uploading files. The agent leverages Google's Gemini language model for natural language understanding and response generation, Groq's API for fast voice transcription, and a custom backend server using the Multi-Capability Protocol (MCP) for executing specific tools and actions.

While the initial vision included building a highly dynamic agent builder for complex ERP systems using only open-source components and advanced RAG, this implementation focuses on a functional Finance Agent showcasing:

*   **Conversational AI:** Interacting with Gemini for understanding requests and generating responses.
*   **Voice Interaction:** Whisper model for speech-to-text.
*   **File Processing:** Uploading documents (CSV, Excel, TXT, JSON) and automatically converting tabular/text data into a structured JSON format stored alongside the original.
*   **Tool Use:** Gemini dynamically calls backend tools (via MCP) to perform actions like retrieving account balances, listing transactions, querying document contents, and processing uploaded files.
*   **Modular Backend:** Using MCP allows the backend logic (tool implementations) to run as a separate process, promoting separation of concerns.

## Features

*   **Web Interface:** Clean chat interface built with FastAPI and HTMX for dynamic updates without full page reloads.
*   **Text & Voice Commands:** Interact via typed messages or by recording voice commands.
*   **Fast Transcription:** Utilizes Groq API for quick and accurate speech-to-text conversion (Whisper-large-v3).
*   **File Upload & Processing:**
    *   Upload CSV, Excel (.xls, .xlsx), JSON, or TXT files.
    *   The backend automatically saves the original file.
    *   **Automated JSON Conversion:** Attempts to convert the *full content* of uploaded CSV, Excel (all sheets), or TXT files into a structured JSON representation (`<original_filename>.json`). This JSON version is saved in the knowledge base alongside the original.
    *   Handles potential decoding errors and imposes size limits for conversion (`MAX_ROWS_FULL_CONVERSION`, `MAX_CHARS_FULL_CONVERSION`).
*   **Gemini Integration:** Uses the `google-generativeai` library to interact with the Gemini family of models (configurable, defaults to 1.5 Flash).
*   **Tool Calling:**
    *   Gemini identifies when a specific capability (tool) defined on the backend server is needed.
    *   The agent logic calls the appropriate tool via MCP (e.g., `get_account_balance`, `list_transactions`, `create_invoice`, `get_document_content`, `process_uploaded_document`).
    *   Results from the tool are sent back to Gemini to inform the final response.
*   **MCP Backend Server:** A separate Python process (`finance_agent_server_json.py`) hosts the tool implementations and manages data persistence (`finance_data.json`) and the document knowledge base (`uploaded_files/`).
*   **Knowledge Base / Basic RAG:**
    *   Uploaded files (originals and generated `.json` versions) are stored in the `uploaded_files/` directory.
    *   `get_document_content` tool allows Gemini to retrieve the full content of any specific file (original or JSON).
    *   `query_knowledge_base` tool performs basic keyword searches across all files in the knowledge base.
*   **Chat History Management:** Maintains conversation history for context and provides a "Clear History" function.

## Architecture

The application consists of three main components:

1.  **FastAPI Web Server (`main.py`):**
    *   Serves the HTML frontend (using Jinja2 templates).
    *   Handles HTTP requests for chat, file uploads, voice transcription, file listing, and history clearing.
    *   Communicates with the `FinanceAgentClientLogic`.
    *   Uses HTMX to update the UI dynamically.
    *   Manages the Groq API client for transcription.

2.  **Agent Client Logic (`finance_agent_logic.py`):**
    *   Acts as the orchestrator between the frontend, the AI model, and the backend server.
    *   Manages the connection to the MCP server using `mcp-fastmcp` client library.
    *   Formats available MCP tools for Gemini.
    *   Manages the conversation history with Gemini.
    *   Handles the interaction flow: sending prompts, processing function calls, calling MCP tools, sending results back.
    *   Specifically handles file uploads by first calling the `process_uploaded_document` tool on the server *before* prompting Gemini, informing the LLM about the newly available original and JSON files.

3.  **MCP Backend Server (`finance_agent_server_json.py`):**
    *   Runs as a separate process, typically managed by the `FinanceAgentClientLogic` via stdio.
    *   Implements the MCP server using `mcp-fastmcp`.
    *   Defines the actual tools (Python functions decorated with `@mcp.tool()`) that perform specific actions (e.g., accessing `finance_data.json`, interacting with files in `uploaded_files/`, converting uploaded files to JSON).
    *   Handles data persistence for basic financial records (`static/finance_data.json`).
    *   Manages the `uploaded_files/` directory.

**Interaction Flow:**

*   **User Input (Text/Voice/File):** User interacts via the web UI. Voice is transcribed via Groq.
*   **FastAPI:** Receives the request. If a file is present, it's read.
*   **Agent Logic:**
    *   Ensures MCP connection is active.
    *   If a file was uploaded, calls the `process_uploaded_document` tool on the MCP server. The server saves the original and attempts to create/save a `.json` version.
    *   Constructs a prompt for Gemini, including the user command and information about any processed file (mentioning the original and the generated `.json` filename).
    *   Sends the prompt and available tools list to Gemini.
*   **Gemini:** Processes the prompt. Either generates a text response or requests a function call (tool use).
*   **Agent Logic:**
    *   If Gemini requests a tool:
        *   Calls the corresponding tool on the MCP server via the MCP protocol.
        *   Receives the result from the MCP server.
        *   Sends the tool result back to Gemini.
        *   Receives the final text response from Gemini.
    *   If Gemini provides a direct text response, uses that.
*   **FastAPI:** Receives the final text response from the Agent Logic, converts it to HTML (using Markdown), and sends it back to the browser via HTMX.

## Technology Stack

*   **Backend Framework:** FastAPI
*   **Web Server:** Uvicorn
*   **Templating:** Jinja2
*   **Frontend Dynamics:** HTMX (via CDN in `index.html`)
*   **AI Model:** Google Gemini (via `google-generativeai`)
*   **Voice Transcription:** Whisper model ( via `Groq API`)
*   **Backend Communication:** MCP (Multi-Capability Protocol) via `mcp-fastmcp`
*   **Data Handling (Server):** Pandas (for CSV/Excel processing)

## Setup and Running

1.  **Prerequisites:**
    *   Python 3.8 or higher
    *   Git

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    *(First, ensure you have the requirements listed. If not, create the file)*
    ```bash
    # If requirements.txt doesn't exist, create it from the installed packages
    # Ensure you have installed: fastapi uvicorn jinja2 python-dotenv google-generativeai mcp-fastmcp groq pandas markdown requests aiohttp # (add any others imported)
    # pip freeze > requirements.txt # Run this after installing needed packages manually if needed

    # Install from requirements.txt
    pip install -r requirements.txt
    ```
    *(Note: You need to create a `requirements.txt` file listing all dependencies, e.g., by running `pip freeze > requirements.txt` after installing them manually.)*

5.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the project root directory.
    *   Add your API keys to the `.env` file:
        ```dotenv
        GEMINI_API_KEY=YOUR_GOOGLE_AI_STUDIO_API_KEY
        GROQ_API_KEY=YOUR_GROQ_CLOUD_API_KEY
        ```
    *   Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/).
    *   Get your Groq API key from [GroqCloud](https://console.groq.com/keys).

6.  **Run the Application:**
    ```bash
    uvicorn main:app --reload --host 127.0.0.1 --port 5001
    ```

7.  **Access the UI:**
    Open your web browser and navigate to `http://127.0.0.1:5001`.

## How It Works: Detailed Flows

### Voice Command

1.  User clicks the microphone button.
2.  Browser records audio (typically webm format).
3.  Audio data is sent to the `/transcribe` endpoint in `main.py`.
4.  `main.py` sends the audio data to the Groq API using the `groq` Python client.
5.  Groq transcribes the audio using Whisper-large-v3.
6.  The transcribed text is returned to the browser and populates the chat input.
7.  User clicks "Send" (or presses Enter).
8.  Follows the standard "Text Command" flow below.

### Text Command / Sending Transcribed Text

1.  User types a command or sends transcribed text.
2.  The command is sent to the `/chat` endpoint in `main.py`.
3.  `main.py` calls `agent_logic_instance.process_single_command`.
4.  `finance_agent_logic` sends the command (and chat history) to Gemini.
5.  Gemini responds with text or a function call request.
6.  If it's a function call (tool use):
    *   `finance_agent_logic` parses the tool name and arguments.
    *   It sends a `call_tool` request to the `finance_agent_server_json` process via MCP.
    *   The server executes the tool function (e.g., `get_account_balance`).
    *   The server returns the result via MCP.
    *   `finance_agent_logic` sends the tool result back to Gemini.
    *   Gemini generates the final text response based on the tool result.
7.  The final text response is sent back through `main.py` to the browser.

### File Upload

1.  User selects a file and optionally types a command.
2.  The file and command are sent to the `/chat` endpoint in `main.py`.
3.  `main.py` reads the file content and passes it along with the command to `agent_logic_instance.process_single_command`.
4.  **Crucially:** `finance_agent_logic` *first* calls the `process_uploaded_document` tool on the MCP server, passing the filename and base64-encoded content.
5.  The `finance_agent_server_json` process:
    *   Decodes the content.
    *   Saves the original file to the `uploaded_files/` directory.
    *   Detects the file type (CSV, Excel, JSON, TXT).
    *   If convertible (CSV, Excel, TXT, valid JSON):
        *   Reads/parses the content (using Pandas for CSV/Excel).
        *   Applies size limits.
        *   Creates a structured JSON representation.
        *   Saves this structure as `<original_filename>.json` in `uploaded_files/`.
    *   Returns a JSON response summarizing the outcome (success, original only, partial failure) including the original filename and the generated JSON filename (if successful).
6.  `finance_agent_logic` receives this summary response from the server.
7.  `finance_agent_logic` constructs the prompt for Gemini:
    *   Includes the summary message from the file processing.
    *   **Explicitly tells Gemini** that the original file (`filename`) and a structured JSON version (`filename.json`) are available in the knowledge base and can be accessed using the `get_document_content` tool.
    *   Includes the user's original text command (if any).
8.  Sends the augmented prompt to Gemini.
9.  The flow continues like a standard text command, but Gemini is now aware of the newly processed file and its JSON counterpart, potentially using `get_document_content` in a subsequent turn if needed.

## File Handling & Knowledge Base

*   **Storage:** All uploaded files and their generated JSON counterparts are stored in the `uploaded_files/` directory relative to the *server script's location* (`finance_agent_server_json.py`).
*   **JSON Conversion:** The server attempts to create a complete JSON representation of CSV, Excel (all sheets), and TXT files. This allows Gemini to potentially analyze the structured data via the `get_document_content` tool applied to the `.json` file.
*   **Access:** The `get_document_content(file_name)` tool can retrieve the content of *any* file in the `uploaded_files/` directory, whether it's an original upload or a server-generated `.json` file.
*   **Querying:** The `query_knowledge_base(query)` tool performs a simple, case-insensitive keyword search across the text content of *all* files in `uploaded_files/`. It returns snippets from files containing the query terms. This is a basic form of RAG.

## Configuration

*   **API Keys:** Configure `GEMINI_API_KEY` and `GROQ_API_KEY` in the `.env` file.
*   **Server Script:** The path to the backend server script is defined in `main.py` (`SERVER_SCRIPT_PATH`).
*   **Upload Directory:** The directory for uploads (`UPLOAD_DIR`) is derived relative to the server script path in both `main.py` and `finance_agent_server_json.py`. Ensure consistency if paths are changed.
*   **Data File:** The path for the simple JSON database is configured in `finance_agent_server_json.py` (`DATA_FILE`).
*   **Conversion Limits:** `MAX_ROWS_FULL_CONVERSION` and `MAX_CHARS_FULL_CONVERSION` in `finance_agent_server_json.py` control the size limits for generating JSON files from large uploads.
*   **Gemini Model:** The Gemini model name can be changed in `finance_agent_logic.py` (`model_name='gemini-1.5-flash-latest'`).

## Adding New Tools/Operations

1.  **Define the Tool Function:** In `finance_agent_server_json.py`, create a new `async` Python function that performs the desired action.
2.  **Decorate the Function:** Add the `@mcp.tool()` decorator above your function definition.
3.  **Add Type Hinting & Docstring:** Use Python type hints for arguments and the return type. Write a clear docstring explaining what the tool does, its arguments, and what it returns. Gemini uses this information to understand how and when to use the tool. The return value should ideally be a JSON string representing the result or status.
4.  **Restart the Application:** When the application restarts, the `FinanceAgentClientLogic` will automatically connect to the MCP server, list the available tools (including your new one), and make it available to Gemini during subsequent interactions.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You should create a LICENSE file with the MIT license text).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.