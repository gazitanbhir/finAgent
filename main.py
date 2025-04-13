# main.py (Corrected for Groq Sync Call)

import asyncio
import os
import sys
from fastapi import FastAPI, Request, HTTPException, Form, Response, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import traceback
import logging
from typing import Optional, List
import html
from dotenv import load_dotenv # Added
import markdown
# --- Groq Imports ---
from groq import Groq, APIError # Added

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv() # Added
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Added

# --- Agent Logic Import ---
try:
    from finance_agent_logic import FinanceAgentClientLogic
except ImportError:
    log.error("FATAL: Could not import FinanceAgentClientLogic.")
    sys.exit(1)


# --- Configuration ---
SERVER_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "finance_agent_server_json.py"))
# Corrected UPLOAD_DIR path finding logic (assuming server script location is stable)
_server_script_dir = os.path.dirname(SERVER_SCRIPT_PATH)
UPLOAD_DIR = os.path.abspath(os.path.join(_server_script_dir, "uploaded_files")) # Path relative to server script

if not os.path.isfile(SERVER_SCRIPT_PATH):
    log.error(f"FATAL ERROR: Cannot find MCP server script at '{SERVER_SCRIPT_PATH}'.")
    sys.exit(1)
else:
     log.info(f"Using MCP server script: {SERVER_SCRIPT_PATH}")

if not os.path.isdir(UPLOAD_DIR):
    log.warning(f"Upload directory derived from server script not found at '{UPLOAD_DIR}'. File listing might fail.")
    # Optionally create it if it should always exist:
    # try:
    #     os.makedirs(UPLOAD_DIR)
    #     log.info(f"Created upload directory: {UPLOAD_DIR}")
    # except OSError as e:
    #     log.error(f"Failed to create upload directory {UPLOAD_DIR}: {e}")


# --- Groq Client Initialization ---
if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY not found in environment variables. Voice transcription will fail.")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        log.info("Groq client initialized successfully.")
    except Exception as e:
        log.error(f"Failed to initialize Groq client: {e}", exc_info=True)
        groq_client = None

# --- FastAPI Setup ---
app = FastAPI(title="AI Finance Agent API")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(static_dir):
    log.warning(f"Static directory not found at '{static_dir}'. Creating.")
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates_root = os.path.dirname(__file__)
templates_dir_check = os.path.join(templates_root, "templates")
if not os.path.isdir(templates_dir_check):
     log.error(f"FATAL: 'templates' directory not found at ({templates_dir_check}).")
     sys.exit(1)
required_templates = ["index.html", "chat_message_fragment.html", "file_list_fragment.html"]
for tmpl in required_templates:
    if not os.path.isfile(os.path.join(templates_dir_check, tmpl)):
        log.error(f"FATAL: Required template '{tmpl}' not found in '{templates_dir_check}'.")
        sys.exit(1)
templates = Jinja2Templates(directory=templates_dir_check)


# --- Global Agent Logic Instance ---
try:
    agent_logic_instance = FinanceAgentClientLogic()
except Exception as init_err:
     log.error(f"FATAL: Failed to initialize FinanceAgentClientLogic: {init_err}")
     traceback.print_exc()
     sys.exit(1)

mcp_connection_lock = asyncio.Lock()

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    log.info("FastAPI starting up...")
    log.info("Attempting initial connection to MCP server...")
    async with mcp_connection_lock:
        try:
            # Check if already connected? Let connect handle it.
            await agent_logic_instance.disconnect()
            await asyncio.sleep(0.1) # Short delay before reconnect
            await agent_logic_instance.connect(SERVER_SCRIPT_PATH)
            if agent_logic_instance.is_connected():
                 log.info("Initial MCP connection successful.")
            else:
                 log.warning("Initial MCP connection attempt finished, but instance is not connected.")
        except ConnectionError as e:
            log.warning(f"Initial connection to MCP server failed on startup: {e}")
        except Exception as e:
            log.error(f"Unexpected error during startup connection: {type(e).__name__}: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    log.info("FastAPI shutting down...")
    log.info("Disconnecting from MCP server...")
    async with mcp_connection_lock:
        try:
            await agent_logic_instance.disconnect()
            log.info("MCP disconnection complete.")
        except Exception as e:
            log.error(f"Error during MCP disconnection on shutdown: {type(e).__name__}: {e}", exc_info=True)

# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    """Serves the main index.html page."""
    log.info("Serving index.html")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/list_files", response_class=HTMLResponse)
async def get_file_list_fragment(request: Request):
    """Returns an HTML fragment containing the list of uploaded files."""
    log.info("Request for file list fragment")
    files = []
    error_message = None
    upload_path = UPLOAD_DIR # Use the globally derived path
    try:
        if not os.path.isdir(upload_path):
            error_message = f"Upload directory not found at configured path: {upload_path}"
            log.error(error_message)
        else:
            raw_files = os.listdir(upload_path)
            files = sorted([f for f in raw_files if os.path.isfile(os.path.join(upload_path, f))])
            log.info(f"Found {len(files)} files in {upload_path}")

    except Exception as e:
        error_message = f"Error listing files in {upload_path}: {e}"
        log.error(error_message, exc_info=True)

    # Render the fragment template
    return templates.TemplateResponse(
        "file_list_fragment.html",
        {"request": request, "files": files, "error_message": error_message}
    )


async def ensure_mcp_connection() -> bool:
    """Ensures the MCP agent is connected, attempting to connect if necessary. Returns True if connected."""
    if agent_logic_instance.is_connected():
        return True
    async with mcp_connection_lock:
        # Double check after acquiring lock
        if agent_logic_instance.is_connected():
            return True
        log.warning("[FastAPI Ensure MCP] MCP connection inactive. Attempting to reconnect...")
        try:
            path_to_use = agent_logic_instance._server_script_path or SERVER_SCRIPT_PATH
            await agent_logic_instance.disconnect() # Clean up potential stale state
            await asyncio.sleep(0.1)
            await agent_logic_instance.connect(path_to_use) # Use stored/config path
            if agent_logic_instance.is_connected():
                log.info("[FastAPI Ensure MCP] MCP Reconnection successful.")
                return True
            else:
                log.error("[FastAPI Error Ensure MCP] Failed to re-establish MCP connection. Check server logs.")
                return False
        except ConnectionError as ce:
            log.error(f"[FastAPI Error Ensure MCP] MCP Reconnection failed: {ce}")
            return False
        except Exception as conn_e:
            log.error(f"[FastAPI Error Ensure MCP] Unexpected error during reconnection attempt: {type(conn_e).__name__}")
            traceback.print_exc()
            return False

# --- NEW Transcription Endpoint ---
@app.post("/transcribe", response_class=JSONResponse)
async def handle_transcription(
    request: Request,
    audio_file: UploadFile = File(...)
):
    """Receives audio data, transcribes using Groq, returns text."""
    log.info(f"Received audio file for transcription: {audio_file.filename} (type: {audio_file.content_type})")

    if not groq_client:
        log.error("Groq client not available (missing API key or initialization failed).")
        raise HTTPException(status_code=503, detail="Transcription service is unavailable.")

    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    try:
        # Read the audio file content - This IS async
        audio_bytes = await audio_file.read()
        log.info(f"Read {len(audio_bytes)} bytes from audio file.")
        if not audio_bytes:
             raise HTTPException(status_code=400, detail="Received empty audio file.")

        # Prepare file data for Groq API
        # Groq SDK expects a file-like object or (filename, file_bytes, content_type) tuple for `file`
        # Using a simple tuple with filename and bytes works reliably.
        file_tuple = (audio_file.filename or "audio.webm", audio_bytes)

        # Call Groq API - Synchronous call, NO await
        log.info("Sending audio to Groq for transcription...")
        transcription = groq_client.audio.transcriptions.create(
            file=file_tuple,
            model="whisper-large-v3", # Or "whisper-large-v3-turbo" if preferred
            response_format="json",  # Ensures response.text is available
        )

        log.info(f"Groq transcription successful. Text length: {len(transcription.text)}")
        # print(transcription.text) # For debugging

        return JSONResponse(content={"transcription": transcription.text})

    except APIError as e:
        # Log the detailed API error from Groq
        log.error(f"Groq API Error during transcription: Status={e.status_code}, Message={e.message}, Body={e.body}", exc_info=False) # Log body for debug
        detail_msg = f"Transcription service API error."
        # Provide a slightly more specific message if possible, but avoid exposing too much internal detail
        if e.status_code == 400:
            detail_msg += " Check audio format or parameters."
        elif e.status_code == 401:
             detail_msg += " Authentication error (check API key)."
        elif e.status_code == 429:
            detail_msg += " Rate limit exceeded."
        elif e.status_code >= 500:
            detail_msg += " Server error on transcription service side."
        raise HTTPException(status_code=502, detail=detail_msg) # 502 Bad Gateway suggests upstream issue
    except Exception as e:
        # Catch-all for other unexpected errors
        log.error(f"Unexpected error during transcription processing: {type(e).__name__}: {e}", exc_info=True)
        error_detail = f"Internal server error during transcription: {type(e).__name__}"
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        try:
             # Closing the file IS async
            await audio_file.close()
        except Exception as close_err:
            log.warning(f"Error closing transcription upload file '{audio_file.filename}': {close_err}")


# --- Existing Chat Endpoint (no changes needed here, copied from previous step) ---
@app.post("/chat", response_class=HTMLResponse)
async def handle_chat_htmx(
    request: Request,
    command: str = Form(''),
    file: Optional[UploadFile] = File(None)
):
    """Handles chat commands and file uploads, returns HTML fragment for HTMX."""
    user_command = command.strip()
    uploaded_file_content: Optional[bytes] = None
    uploaded_filename: Optional[str] = None
    user_message_display = user_command
    file_processed_trigger = False

    # --- File handling logic (as before) ---
    if file:
        log.info(f"[FastAPI /chat] Received file: '{file.filename}' (Type: {file.content_type})")
        MAX_FILE_SIZE = 50 * 1024 * 1024 # 50MB limit
        actual_size = 0
        try:
            uploaded_file_content = await file.read()
            actual_size = len(uploaded_file_content)
            uploaded_filename = file.filename
            log.info(f"[FastAPI /chat] Read {actual_size} bytes from '{uploaded_filename}'")

            if actual_size > MAX_FILE_SIZE:
                log.error(f"Uploaded file '{file.filename}' exceeds size limit ({actual_size} > {MAX_FILE_SIZE}).")
                error_html = templates.TemplateResponse("chat_message_fragment.html", {
                    "request": request, "sender": "error", "message": f"File exceeds maximum size limit of {MAX_FILE_SIZE // 1024 // 1024}MB."
                }).body.decode('utf-8')
                return HTMLResponse(content=error_html, status_code=413)

            if not uploaded_file_content:
                 log.warning(f"Uploaded file '{file.filename}' is empty.")

            # Use html.escape for safety when displaying filename
            user_message_display += f" (File: {html.escape(uploaded_filename or 'unknown')})"

        except Exception as read_err:
             log.error(f"Error reading uploaded file '{file.filename}': {read_err}", exc_info=True)
             error_html = templates.TemplateResponse("chat_message_fragment.html", {
                    "request": request, "sender": "error", "message": f"Could not read uploaded file: {read_err}"
             }).body.decode('utf-8')
             return HTMLResponse(content=error_html, status_code=400)
        finally:
             try:
                 await file.close()
             except Exception as close_err:
                 log.warning(f"Error closing uploaded file '{file.filename}': {close_err}")
    # --- End File handling ---


    if not user_command and not uploaded_file_content:
        log.warning("[FastAPI /chat] Received empty command and no file.")
        if not user_message_display:
             return HTMLResponse(content="", status_code=204) # No command, no file

    log_msg = f"[FastAPI /chat] Request: Command='{user_command[:100]}{'...' if len(user_command)>100 else ''}'"
    if uploaded_filename:
        log_msg += f", File='{uploaded_filename}' ({len(uploaded_file_content) if uploaded_file_content else 0} bytes)"
    log.info(log_msg)

    # Ensure MCP Connection
    is_connected = await ensure_mcp_connection()
    if not is_connected:
         log.error("[FastAPI /chat] Aborting - MCP connection failed or unavailable.")
         error_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "error", "message": "Agent backend unavailable. Could not connect."
         }).body.decode('utf-8')
         user_msg_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "user", "message": user_message_display
         }).body.decode('utf-8')
         return HTMLResponse(content=user_msg_html + error_html, status_code=503)

    # Call Agent Logic
    agent_response_text = "[Error] Agent failed to process command."
    agent_response_html = agent_response_text  # fallback in case of error
    try:
        agent_response_text = await agent_logic_instance.process_single_command(
            user_command,
            approve_tool_call=True,
            uploaded_file_content=uploaded_file_content,
            uploaded_filename=uploaded_filename
        )

        # Markdown to HTML
        agent_response_html = markdown.markdown(agent_response_text, extensions=["extra", "sane_lists"])

        # Trigger filesUpdated if file processed successfully
        if uploaded_filename and (
            "[File Processing Result" in agent_response_text or
            "[File Processing Info" in agent_response_text
        ):
            if "error" not in agent_response_text.lower() and "failed" not in agent_response_text.lower():
                file_processed_trigger = True

    except ConnectionError as ce:
        log.error(f"[FastAPI Error] MCP Connection error during processing: {ce}", exc_info=True)
        agent_response_text = f"[Error] Agent backend connection lost during processing: {ce}"
        user_msg_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "user", "message": user_message_display
         }).body.decode('utf-8')
        error_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "error", "message": agent_response_text
         }).body.decode('utf-8')
        return HTMLResponse(content=user_msg_html + error_html, status_code=503)
    except HTTPException as he:
         log.error(f"[FastAPI Error] HTTP Exception during processing: {he.status_code} - {he.detail}", exc_info=True)
         agent_response_text = f"[Error {he.status_code}] {he.detail}"
         user_msg_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "user", "message": user_message_display
         }).body.decode('utf-8')
         error_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "error", "message": agent_response_text
         }).body.decode('utf-8')
         return HTMLResponse(content=user_msg_html + error_html, status_code=he.status_code)
    except Exception as e:
        log.error(f"[FastAPI Error] Unexpected error processing chat: {type(e).__name__}: {e}", exc_info=True)
        agent_response_text = f"[Error] An unexpected server error occurred: {type(e).__name__}"
        user_msg_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "user", "message": user_message_display
         }).body.decode('utf-8')
        error_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "error", "message": agent_response_text
         }).body.decode('utf-8')
        return HTMLResponse(content=user_msg_html + error_html, status_code=500)

    # Prepare HTML Fragment Response
    combined_html = templates.TemplateResponse("chat_message_fragment.html", {
        "request": request, "sender": "user", "message": user_message_display
    }).body.decode('utf-8')

    agent_sender_type = "agent"
    if agent_response_text.startswith(("[Error", "[Client Logic Error]")):
        agent_sender_type = "error"
    elif agent_response_text.startswith(("[File Processing Info", "[Client Logic Info]", "[File Processing Result")):
        agent_sender_type = "info"

    combined_html += templates.TemplateResponse("chat_message_fragment.html", {
    "request": request, "sender": agent_sender_type, "message": agent_response_html  # now HTML
    }).body.decode('utf-8')


    log.info(f"[FastAPI /chat] Sending HTML fragment response (Agent: {agent_sender_type})")

    headers = {}
    if file_processed_trigger:
        headers["HX-Trigger"] = "filesUpdated"
        log.info("[FastAPI /chat] Adding HX-Trigger: filesUpdated header")

    return HTMLResponse(content=combined_html, headers=headers)


@app.post("/clear_history", response_class=HTMLResponse)
async def clear_history_htmx(request: Request):
    """Clears agent history and returns HTML fragment for chatbox replacement."""
    log.info("[FastAPI /clear_history] Request received.")
    try:
        agent_logic_instance.clear_chat_history()
        log.info("[FastAPI /clear_history] Agent history cleared.")

        welcome_message_html = templates.TemplateResponse("chat_message_fragment.html", {
            "request": request, "sender": "agent welcome", "message": "Hello! History cleared. How can I help?"
        }).body.decode('utf-8')

        return HTMLResponse(content=welcome_message_html)

    except Exception as e:
         log.error(f"[FastAPI /clear_history] Error clearing history: {type(e).__name__}: {e}", exc_info=True)
         error_html = templates.TemplateResponse("chat_message_fragment.html", {
             "request": request, "sender": "error", "message": "Failed to clear agent history on backend."
         }).body.decode('utf-8')
         return HTMLResponse(content=error_html, status_code=500)

# --- Run Command ---
if __name__ == "__main__":
    import uvicorn
    log.info("-----------------------------------------------------")
    log.info("Starting AI Finance Agent FastAPI Application (HTMX + Voice)...")
    log.info(f"MCP Server Script: {SERVER_SCRIPT_PATH}")
    log.info(f"Upload Directory: {UPLOAD_DIR}")
    
    # Render-specific modifications
    host = "0.0.0.0" if os.getenv("RENDER") else "127.0.0.1"
    port = int(os.getenv("PORT", "5001"))
    reload = not bool(os.getenv("RENDER"))  # Disable reload in production
    
    log.info(f"Run with: uvicorn main:app --host {host} --port {port} {'--reload' if reload else ''}")
    log.info("-----------------------------------------------------")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )