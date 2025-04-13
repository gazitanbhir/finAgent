# finance_agent_logic.py (Handles JSON file creation confirmation)

import asyncio
import os
import sys
import json
import base64
from contextlib import AsyncExitStack
from dotenv import load_dotenv
import traceback
from typing import List, Optional, Dict, Any

import google.generativeai as genai
from google.generativeai import types as genai_types
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
import google.generativeai.protos as protos

from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file or environment variables")

genai.configure(api_key=GEMINI_API_KEY)

try:
    gemini_model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest', # Or 'gemini-1.5-pro-latest'
    )
    print("[Client Logic] Gemini model initialized.")
except Exception as e:
    print(f"[Client Logic FATAL] Error initializing Gemini model: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise

# --- Safety Settings ---
SAFETY_SETTINGS_CONFIG = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
# --- End Safety Settings ---


# --- MCP Client Logic Class ---
class FinanceAgentClientLogic:
    def __init__(self):
        self.mcp_session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._connection_task: Optional[asyncio.Task] = None
        self._server_script_path: Optional[str] = None
        self._is_connecting: bool = False
        self._connect_lock = asyncio.Lock()

        self.gemini_tool_objects: List[protos.Tool] = []
        self.chat_history: List[protos.Content] = []
        print("[Client Logic] Instance created.")

    # --- connect, disconnect, _cleanup_connection_resources, is_connected, _format_mcp_tools_for_gemini ---
    # (Keep the existing connect, disconnect, _cleanup_connection_resources, is_connected,
    #  _format_mcp_tools_for_gemini functions as they were)
    # ... (connect, disconnect, _cleanup_connection_resources, is_connected code remains here) ...
    # ... (_format_mcp_tools_for_gemini code remains here) ...
    async def connect(self, server_script_path: str):
        """Establishes connection to the MCP server process."""
        async with self._connect_lock:
            if self.is_connected() or self._is_connecting:
                print(f"[Client Logic] Connect called but already {'connected' if self.is_connected() else 'connecting'}.")
                return

            print(f"[Client Logic] Attempting to connect to MCP server: {server_script_path}...")
            self._is_connecting = True
            self._server_script_path = server_script_path # Store for potential reconnect

            try:
                self._exit_stack = AsyncExitStack() # Create new stack for this connection attempt
                server_params = StdioServerParameters(
                    command=sys.executable, # Use the current python interpreter
                    args=[server_script_path],
                )
                stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = stdio_transport
                self.mcp_session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

                print("[Client Logic] MCP transport connected, initializing session...")
                await self.mcp_session.initialize()
                print("[Client Logic] MCP Session Initialized.")

                # Fetch tools after successful initialization
                mcp_tool_list = await self.mcp_session.list_tools()
                self.gemini_tool_objects = self._format_mcp_tools_for_gemini(mcp_tool_list.tools)
                tool_names = [decl.name for tool in self.gemini_tool_objects for decl in tool.function_declarations]
                print(f"[Client Logic] Connection successful. Available tools: {tool_names}")
                self._is_connecting = False

            except ConnectionRefusedError as e:
                print(f"[Client Logic Error] Connection refused: {e}. Is the server script path correct and executable?")
                await self._cleanup_connection_resources()
                self._is_connecting = False
                raise ConnectionError(f"Connection refused: {e}") from e
            except FileNotFoundError as e:
                print(f"[Client Logic Error] Server script not found at '{server_script_path}': {e}")
                await self._cleanup_connection_resources()
                self._is_connecting = False
                raise ConnectionError(f"Server script not found: {server_script_path}") from e
            except mcp_types.FatalError as e:
                print(f"[Client Logic Error] MCP Fatal Error during connection: {e}")
                await self._cleanup_connection_resources()
                self._is_connecting = False
                raise ConnectionError(f"MCP Fatal Error during connection: {e}") from e
            except TimeoutError:
                print("[Client Logic Error] Timeout during MCP connection/initialization.")
                await self._cleanup_connection_resources()
                self._is_connecting = False
                raise ConnectionError("Timeout during MCP connection")
            except Exception as e:
                print(f"[Client Logic Error] Unexpected error during connection: {type(e).__name__}: {e}")
                traceback.print_exc()
                await self._cleanup_connection_resources()
                self._is_connecting = False
                # Raise as ConnectionError to signal failure to connect
                raise ConnectionError(f"Unexpected error during connection: {type(e).__name__}: {e}") from e

    async def disconnect(self):
        """Disconnects from the MCP server and cleans up resources."""
        async with self._connect_lock:
            if not self.is_connected() and not self._is_connecting:
                print("[Client Logic] Disconnect called but not connected or connecting.")
                return

            print("[Client Logic] Disconnecting from MCP server...")
            self._is_connecting = False # Ensure connecting flag is false
            await self._cleanup_connection_resources()
            print("[Client Logic] MCP Disconnection complete.")

    async def _cleanup_connection_resources(self):
        """Internal helper to clean up session and exit stack."""
        if self.mcp_session:
            try:
                if hasattr(self.mcp_session, 'close_gracefully'):
                     await self.mcp_session.close_gracefully(timeout=2.0)
                elif hasattr(self.mcp_session, 'aclose'): # Fallback? Check ClientSession API
                     await self.mcp_session.aclose()
            except Exception as close_err:
                print(f"[Client Logic Warning] Error during MCP session close: {close_err}")
            finally:
                self.mcp_session = None

        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as stack_err:
                print(f"[Client Logic Warning] Error closing AsyncExitStack: {stack_err}")
            finally:
                self._exit_stack = None
        self.gemini_tool_objects = [] # Clear tools on disconnect

    def is_connected(self) -> bool:
            """Checks if the MCP session object exists."""
            return self.mcp_session is not None

    def _format_mcp_tools_for_gemini(self, mcp_tools: List[mcp_types.Tool]) -> List[protos.Tool]:
        """Converts MCP Tool definitions into Gemini protos.Tool format."""
        gemini_tools_list = []
        if not mcp_tools:
            print("[Client Logic Warning] No tools received from MCP server.")
            return []

        print(f"[Client Logic] Formatting {len(mcp_tools)} MCP tools for Gemini...")
        for tool in mcp_tools:
            if not isinstance(tool, mcp_types.Tool):
                print(f"[Client Logic Warning] Skipping invalid item in tool list: {type(tool)}")
                continue

            properties = {}
            input_schema = getattr(tool, 'inputSchema', None)
            schema_props = {}
            required_list = []
            if isinstance(input_schema, dict):
                schema_props = input_schema.get("properties", {})
                req = input_schema.get("required", [])
                required_list = [item for item in req if isinstance(item, str)] if isinstance(req, list) else []

            if isinstance(schema_props, dict):
                for name, schema in schema_props.items():
                    if not isinstance(schema, dict):
                        print(f"[Client Logic Warning] Skipping invalid property schema for '{name}' in tool '{tool.name}': not a dict.")
                        continue
                    prop_type_str = schema.get("type", "string").upper()
                    prop_description = schema.get("description", "")
                    prop_type_enum = protos.Type.TYPE_UNSPECIFIED

                    # Mapping from JSON Schema types to Gemini protos.Type
                    type_map = {
                        "NUMBER": protos.Type.NUMBER, "INTEGER": protos.Type.INTEGER,
                        "BOOLEAN": protos.Type.BOOLEAN, "ARRAY": protos.Type.ARRAY,
                        "OBJECT": protos.Type.OBJECT, "STRING": protos.Type.STRING,
                    }
                    prop_type_enum = type_map.get(prop_type_str, protos.Type.STRING)
                    if prop_type_enum == protos.Type.STRING and prop_type_str != "STRING":
                        print(f"[Client Logic Warning] Unknown property type '{prop_type_str}' for '{name}' in tool '{tool.name}'. Defaulting to STRING.")

                    properties[name] = protos.Schema(type=prop_type_enum, description=prop_description)
            elif input_schema and "properties" in input_schema:
                 print(f"[Client Logic Warning] 'properties' for tool '{tool.name}' is not a dictionary, skipping properties.")

            tool_description = getattr(tool, 'description', "")
            if not isinstance(tool_description, str): tool_description = ""

            func_decl = protos.FunctionDeclaration(
                name=tool.name,
                description=tool_description,
                parameters=protos.Schema(
                    type=protos.Type.OBJECT,
                    properties=properties,
                    required=required_list,
                 )
            )
            gemini_tools_list.append(protos.Tool(function_declarations=[func_decl]))
            print(f"[Client Logic] Formatted tool: {tool.name} with params: {list(properties.keys())}, required: {required_list}")

        print(f"[Client Logic] Finished formatting tools. Total: {len(gemini_tools_list)}")
        return gemini_tools_list

    # --- MODIFIED: Only extracts summary message ---
    async def _call_file_processing_tool(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        Internal helper to call the server's file processing tool.
        Returns a dictionary containing the summary message and the name of the
        generated JSON file (if created).
        """
        result = {
            "summary": f"[Error: Could not invoke processing for file '{filename}'.]",
            "generated_json_filename": None,
            "status": "error"
        }
        if not self.is_connected() or not self.mcp_session:
             raise ConnectionError("MCP session lost before file processing tool call.")

        try:
            print(f"[Client Logic] Calling 'process_uploaded_document' tool for '{filename}'...")
            content_b64 = base64.b64encode(content).decode('utf-8')

            tool_result: mcp_types.CallToolResult = await self.mcp_session.call_tool(
                name="process_uploaded_document",
                arguments={"filename": filename, "content_base64": content_b64}
            )

            # Extract summary and generated filename from the result
            if tool_result.content and hasattr(tool_result.content[0], 'text') and tool_result.content[0].text:
                try:
                    result_data = json.loads(tool_result.content[0].text)
                    result["summary"] = result_data.get("message", f"[File '{filename}' processed, no summary returned]")
                    result["status"] = result_data.get("status", "unknown")
                    result["generated_json_filename"] = result_data.get("generated_json_filename") # May be None

                    print(f"[Client Logic] File processing status: {result['status']}, Generated JSON: {result['generated_json_filename']}")

                    # Format summary for user clarity based on status
                    if result["status"] == "error":
                        result["summary"] = f"[File Processing Error for '{filename}': {result['summary']}]"
                    elif result["status"] == "partial_failure_conversion":
                         result["summary"] = f"[File Processing Warning for '{filename}': {result['summary']}]"
                    elif result["status"] == "success_original_only":
                         result["summary"] = f"[File Processing Info for '{filename}': {result['summary']}]"
                    else: # success
                         result["summary"] = f"[File Processing Result for '{filename}': {result['summary']}]"

                except json.JSONDecodeError:
                    error_msg = f"[File '{filename}' processed, but result format invalid JSON: {tool_result.content[0].text[:100]}...]"
                    print(f"[Client Logic Warning] {error_msg}")
                    result["summary"] = error_msg
                    result["status"] = "error"
                except Exception as parse_e:
                    error_msg = f"[Error parsing file processing result for '{filename}': {parse_e}]"
                    print(f"[Client Logic Error] {error_msg}")
                    result["summary"] = error_msg
                    result["status"] = "error"

            elif tool_result.error:
                error_msg = f"[Error processing file '{filename}' via tool: {tool_result.error}]"
                print(f"[Client Logic Error] {error_msg}")
                result["summary"] = error_msg
                result["status"] = "error"
            else:
                 warn_msg = f"[File '{filename}' sent, but no content/error returned by the tool.]"
                 print(f"[Client Logic Warning] {warn_msg}")
                 result["summary"] = warn_msg
                 result["status"] = "unknown"

            print(f"[Client Logic] File processing summary: {result['summary'][:150]}...")
            return result # Return the dictionary

        except ConnectionError:
            print(f"[Client Logic Error] Connection lost during file processing tool call.")
            raise # Re-raise ConnectionError
        except mcp_types.ToolNotFoundError:
            error_msg = f"[Error: Tool 'process_uploaded_document' not available.]"
            print(f"[Client Logic Error] {error_msg}")
            result["summary"] = error_msg
            result["status"] = "error"
            return result # Return error dict
        except Exception as proc_err:
            error_msg = f"[Error: Client-side issue calling processing tool for '{filename}'.]"
            print(f"[Client Logic Error] Unexpected error calling file processing tool: {type(proc_err).__name__}: {proc_err}")
            traceback.print_exc()
            result["summary"] = error_msg
            result["status"] = "error"
            return result # Return error dict

    # --- MODIFIED: Informs LLM about the generated JSON file ---
    async def process_single_command(
        self,
        command: str,
        approve_tool_call: bool = True,
        uploaded_file_content: Optional[bytes] = None,
        uploaded_filename: Optional[str] = None
    ) -> str:
        """
        Processes a user command (and optional uploaded file), interacts with Gemini, handles tool calls.
        If a file is uploaded, it calls a server tool to process it and save a .json version.
        The prompt informs Gemini that the .json file is available via 'get_document_content'.
        """
        if not self.is_connected() or not self.mcp_session:
            print("[Client Logic Error] Cannot process command: Not connected to MCP server.")
            raise ConnectionError("MCP session is not active.")

        original_command = command
        processing_summary_msg = ""
        generated_json_filename = None

        # --- Step 1: Process Uploaded File (if any) ---
        if uploaded_file_content and uploaded_filename:
            try:
                processing_result = await self._call_file_processing_tool(
                    uploaded_filename,
                    uploaded_file_content
                )
                processing_summary_msg = processing_result.get("summary", "")
                generated_json_filename = processing_result.get("generated_json_filename")

            except ConnectionError as ce:
                return f"[Client Logic Error] Connection failed during file processing: {ce}"
            except Exception as e:
                 return f"[Client Logic Error] Unexpected error during file processing call: {e}"

            # Construct the prompt for Gemini
            prompt_parts = []
            if processing_summary_msg:
                prompt_parts.append(f"File Processing Note: {processing_summary_msg}")

            # Inform Gemini about the available JSON file
            if generated_json_filename:
                 prompt_parts.append(f"\nA structured JSON version of the uploaded file ('{uploaded_filename}') has been saved as '{generated_json_filename}' in the knowledge base.")
                 prompt_parts.append(f"If you need to analyze the full content, use the 'get_document_content' tool with the filename '{generated_json_filename}'.")
            else:
                prompt_parts.append(f"\n[Info: A structured JSON version could not be created or saved for '{uploaded_filename}'. You can try analyzing the original using 'get_document_content' if it's text-based, or use other tools.]")


            if original_command:
                 prompt_parts.append(f"\n\nUser Command: {original_command}")
            else:
                 # If only a file was uploaded, ask Gemini what to do next
                 prompt_parts.append(f"\n\nUser Command: The file '{uploaded_filename}' has been processed as noted above. What would you like to do next? (e.g., summarize the JSON version, ask questions about it)")

            command = "\n".join(prompt_parts) # Update the command string for Gemini

        # --- Step 2: Interact with Gemini ---
        print(f"[Client Logic] Processing command for Gemini: {command[:250]}{'...' if len(command)>250 else ''}")
        if not command.strip():
             print("[Client Logic Warning] Command became empty, returning summary or info.")
             return processing_summary_msg if processing_summary_msg else "[Client Logic Info] File processed, but no further command resulted."

        user_message = protos.Content(role="user", parts=[protos.Part(text=command)])
        current_chat_history = self.chat_history + [user_message]

        try:
            print(f"[Client Logic] Sending to Gemini. History: {len(current_chat_history)}, Tools: {len(self.gemini_tool_objects)}")
            response = await gemini_model.generate_content_async(
                contents=current_chat_history,
                tools=self.gemini_tool_objects,
                safety_settings=SAFETY_SETTINGS_CONFIG,
            )

            # --- Process Gemini Response (Safety, Candidates, Function Calls, Text) ---
            # (This part remains largely the same as the previous version)
            # ... (Safety/Error Checks - Prompt Feedback, Candidates) ...
            prompt_feedback = getattr(response, 'prompt_feedback', None)
            if prompt_feedback and getattr(prompt_feedback, 'block_reason', None):
                block_reason = getattr(prompt_feedback, 'block_reason', 'N/A')
                safety_ratings_str = str(getattr(prompt_feedback, 'safety_ratings', 'N/A'))
                error_msg = f"[Client Logic Error] Gemini request blocked (prompt). Reason: {block_reason}, Safety: {safety_ratings_str}"
                print(error_msg)
                return error_msg

            if not response.candidates:
                finish_reason = str(getattr(response, 'finish_reason', 'Unknown (No Candidates)'))
                safety_ratings_str = str(getattr(prompt_feedback, 'safety_ratings', 'N/A')) if prompt_feedback else 'N/A'
                error_msg = f"[Client Logic Error] Gemini response missing candidates. Finish Reason: {finish_reason}, Safety: {safety_ratings_str}"
                print(error_msg)
                return error_msg

            # --- Process Candidate Content ---
            candidate = response.candidates[0]
            response_content = candidate.content

            # Add user message and model's initial response to history
            self.chat_history.append(user_message)
            self.chat_history.append(response_content)

            # --- Safety Check (Candidate Finish Reason & Safety Ratings) ---
            finish_reason = getattr(candidate, 'finish_reason', None)
            allowed_reasons = ('STOP', 'MAX_TOKENS', 'FUNCTION_CALL', 'FINISH_REASON_UNSPECIFIED')
            if finish_reason and finish_reason.name not in allowed_reasons:
                 safety_ratings = getattr(candidate, 'safety_ratings', 'N/A')
                 error_msg = f"[Client Logic Error] Gemini response stopped unexpectedly. Finish Reason: {finish_reason.name}, Safety: {safety_ratings}"
                 print(error_msg)
                 if len(self.chat_history) >= 2: self.chat_history = self.chat_history[:-2] # Pop user, model
                 return error_msg

            # --- Function Call Check ---
            function_call = None
            for part in response_content.parts:
                if hasattr(part, 'function_call') and getattr(part.function_call, 'name', None):
                    function_call = part.function_call
                    break

            if function_call:
                tool_name = function_call.name
                tool_args = dict(function_call.args) if hasattr(function_call, 'args') else {}
                print(f"[Client Logic] Gemini requests tool call: {tool_name}(Args: {json.dumps(tool_args)})")

                # --- Execute Tool via MCP ---
                if not approve_tool_call:
                     # Handle denied tool call
                     tool_response_part = protos.Part(
                         function_response=protos.FunctionResponse(
                             name=tool_name,
                             response={"result": "Tool call denied by user configuration."}
                         )
                     )
                else:
                    print(f"[Client Logic] Executing tool '{tool_name}' via MCP...")
                    try:
                        if not self.is_connected() or not self.mcp_session:
                             raise ConnectionError("MCP session lost before tool execution.")

                        tool_result: mcp_types.CallToolResult = await self.mcp_session.call_tool(
                            name=tool_name,
                            arguments=tool_args
                        )
                        print(f"[Client Logic] MCP tool '{tool_name}' raw result: {tool_result}")

                        # Process Tool Result for Gemini
                        result_content_struct = {"result": f"Tool '{tool_name}' executed."}
                        if tool_result.content and isinstance(tool_result.content, list) and tool_result.content:
                            first_part = tool_result.content[0]
                            if hasattr(first_part, 'text') and first_part.text is not None:
                                try:
                                    parsed_json = json.loads(first_part.text)
                                    if isinstance(parsed_json, dict):
                                        result_content_struct = parsed_json
                                    else:
                                         result_content_struct = {"result": parsed_json}
                                except json.JSONDecodeError:
                                    print(f"[Client Logic Warning] Tool '{tool_name}' result not JSON. Returning as text.")
                                    result_content_struct = {"result": first_part.text}
                                except Exception as parse_e:
                                     print(f"[Client Logic Error] Error processing tool '{tool_name}' result text: {parse_e}")
                                     result_content_struct = {"error": f"Error parsing tool result: {parse_e}"}
                            else:
                                 result_content_struct = {"result": f"Tool executed, but returned unexpected/empty content type: {type(first_part).__name__}"}
                        elif tool_result.error:
                             print(f"[Client Logic Warning] MCP tool '{tool_name}' returned an error: {tool_result.error}")
                             result_content_struct = {"error": str(tool_result.error)}
                        else:
                             print(f"[Client Logic Warning] Tool '{tool_name}' executed but returned no content or error.")

                        print(f"[Client Logic] Prepared response for Gemini from '{tool_name}': {result_content_struct}")
                        tool_response_part = protos.Part(
                            function_response=protos.FunctionResponse(
                                name=tool_name,
                                response=result_content_struct
                            )
                        )
                    except ConnectionError as ce:
                        print(f"[Client Logic Error] Connection lost during tool execution: {ce}")
                        if len(self.chat_history) >= 2: self.chat_history = self.chat_history[:-2] # user, model
                        raise # Re-raise ConnectionError for FastAPI
                    except Exception as tool_error:
                        print(f"[Client Logic Error] Error calling/processing MCP tool '{tool_name}': {type(tool_error).__name__}: {tool_error}")
                        traceback.print_exc()
                        tool_response_part = protos.Part(
                           function_response=protos.FunctionResponse(
                               name=tool_name,
                               response={"error": f"Client-side error executing tool '{tool_name}': {type(tool_error).__name__}: {tool_error}"}
                           )
                        )

                # --- Send Tool Response back to Gemini ---
                tool_response_message = protos.Content(role="function", parts=[tool_response_part])
                self.chat_history.append(tool_response_message)
                current_chat_history_for_final = self.chat_history

                print(f"[Client Logic] Sending tool response for '{tool_name}' back to Gemini. History: {len(current_chat_history_for_final)}")
                final_response = await gemini_model.generate_content_async(
                    contents=current_chat_history_for_final,
                    tools=self.gemini_tool_objects,
                    safety_settings=SAFETY_SETTINGS_CONFIG
                )

                # --- Process Final Gemini Response ---
                # (Standard checks remain the same)
                final_prompt_feedback = getattr(final_response, 'prompt_feedback', None)
                if final_prompt_feedback and getattr(final_prompt_feedback, 'block_reason', None):
                    block_reason = getattr(final_prompt_feedback, 'block_reason', 'N/A')
                    error_msg = f"[Client Logic Error] Gemini request *after tool call* blocked. Reason: {block_reason}"
                    print(error_msg)
                    if len(self.chat_history) >= 3: self.chat_history = self.chat_history[:-3]
                    return error_msg

                if not final_response.candidates:
                    error_msg = f"[Client Logic Error] Gemini response *after tool call* missing candidates."
                    print(error_msg)
                    if len(self.chat_history) >= 3: self.chat_history = self.chat_history[:-3]
                    return error_msg

                final_candidate = final_response.candidates[0]
                final_response_content = final_candidate.content

                final_finish_reason = getattr(final_candidate, 'finish_reason', None)
                final_allowed_reasons = ('STOP', 'MAX_TOKENS', 'FINISH_REASON_UNSPECIFIED')
                if final_finish_reason and final_finish_reason.name not in final_allowed_reasons:
                     error_msg = f"[Client Logic Error] Gemini final response stopped unexpectedly. Reason: {final_finish_reason.name}"
                     print(error_msg)
                     if len(self.chat_history) >= 3: self.chat_history = self.chat_history[:-3]
                     return error_msg

                # Append final model response
                self.chat_history.append(final_response_content)

                # Extract text
                final_text = "".join(part.text for part in final_response_content.parts if hasattr(part, 'text') and part.text).strip()

                if final_text:
                     print(f"[Client Logic] Gemini final response after tool call: {final_text[:100]}...")
                     return final_text
                else:
                     print("[Client Logic Warning] Gemini final response after tool call did not contain text.")
                     return "[Client Logic Info] Agent processed the tool request but provided no final text response."

            else:
                # --- No function call, direct response ---
                direct_text = "".join(part.text for part in response_content.parts if hasattr(part, 'text') and part.text).strip()

                if direct_text:
                    print(f"[Client Logic] Gemini direct response: {direct_text[:100]}...")
                    return direct_text
                else:
                    finish_reason_name = getattr(getattr(candidate, 'finish_reason', None), 'name', 'N/A')
                    print(f"[Client Logic Warning] Gemini direct response contained no text. Finish Reason: {finish_reason_name}")
                    if finish_reason_name not in ('STOP', 'MAX_TOKENS', 'FINISH_REASON_UNSPECIFIED'):
                        if len(self.chat_history) >= 2: self.chat_history = self.chat_history[:-2]
                        return f"[Client Logic Error] Agent provided an empty response with finish reason: {finish_reason_name}."
                    else:
                         return "[Client Logic Info] Agent provided no text in response."

        # --- Exception Handling ---
        # (Keep existing exception handlers) ...
        except genai_types.BlockedPromptException as bpe:
             print(f"[Client Logic Error] Gemini request blocked (BlockedPromptException): {bpe}")
             error_msg = f"[Client Logic Error] Gemini request blocked. Content may violate safety policies."
             # Don't modify history here as nothing was added for this failed turn
             return error_msg
        except genai_types.StopCandidateException as sce:
             print(f"[Client Logic Error] Gemini candidate stopped unexpectedly (StopCandidateException): {sce}")
             error_msg = f"[Client Logic Error] Gemini response generation stopped unexpectedly. Reason: {sce}"
             # Clean up history potentially added before the exception
             if len(self.chat_history) >= 2 and self.chat_history[-2] is user_message:
                 self.chat_history = self.chat_history[:-2] # Pop user, model response attempt
             elif len(self.chat_history) >= 1 and self.chat_history[-1] is user_message:
                  self.chat_history.pop() # Pop only user message
             return error_msg
        except genai_types.GoogleAPIError as api_error:
             print(f"[Client Logic Error] Google API Error: {type(api_error).__name__}: {api_error}")
             traceback.print_exc()
             if self.chat_history and self.chat_history[-1] is user_message: self.chat_history.pop()
             return f"[Client Logic Error] An error occurred with the AI service: {api_error}. Check API key and quota."
        except ConnectionError:
             print("[Client Logic Error] ConnectionError during processing.")
             if self.chat_history and self.chat_history[-1] is user_message: self.chat_history.pop()
             raise # Re-raise for FastAPI
        except Exception as e:
            print(f"[Client Logic Error] Unexpected error during command processing: {type(e).__name__}: {e}")
            traceback.print_exc()
            if self.chat_history and self.chat_history[-1] is user_message: self.chat_history.pop()
            return f"[Client Logic Error] An unexpected error occurred ({type(e).__name__}). Please check server logs."

    def clear_chat_history(self):
        """Clears the conversation history."""
        print("[Client Logic] Clearing chat history.")
        self.chat_history = []

# --- Direct Test Function (Optional: Update if needed) ---
# async def run_direct_test(): ...

if __name__ == "__main__":
    print("Run main.py with uvicorn to start the web application.")