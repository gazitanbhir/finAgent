# finance_agent_server_json.py
# MCP Server - Enhanced: Saves full JSON conversion of uploads

import asyncio
from mcp.server.fastmcp import FastMCP
import json
import datetime
import os
import uuid
import base64
import csv
import io
from typing import Any, Dict, List, Optional
import traceback
import pandas as pd # For Excel/CSV reading
# Ensure openpyxl is installed for .xlsx: pip install openpyxl

# --- Configuration ---
DATA_FILE = "static/finance_data.json"
UPLOAD_DIR = "uploaded_files" # Directory for RAG documents AND generated JSONs

# Memory/Performance limits for full conversion (adjust as needed)
MAX_ROWS_FULL_CONVERSION = 50000  # Max rows for CSV/Excel JSON file
MAX_CHARS_FULL_CONVERSION = 5 * 1024 * 1024 # Max 5MB chars for TXT JSON file

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print(f"[Server] Created upload directory: '{UPLOAD_DIR}'")

# --- Data Persistence (load_data, save_data, save_data_async) ---
# (Keep the existing load_data, save_data, _save_lock, save_data_async functions)
# ... (load_data, save_data, _save_lock, save_data_async code remains here) ...
def load_data() -> dict[str, Any]:
    """Loads data from the JSON file."""
    default_data = {"accounts": {}, "transactions": [], "invoices": {}, "budgets": {}, "vendors": {}}
    if not os.path.exists(DATA_FILE):
        print(f"[Server] Data file '{DATA_FILE}' not found. Initializing.")
        save_data(default_data) # Use sync save here as it's startup
        return default_data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                print(f"[Server] Data file '{DATA_FILE}' is empty. Initializing.")
                save_data(default_data) # Sync save
                return default_data
            loaded = json.loads(content)
            updated = False
            for key, value in default_data.items():
                if key not in loaded:
                    loaded[key] = value
                    updated = True
            if updated:
                print(f"[Server] Data file '{DATA_FILE}' missing keys. Updated and saving.")
                save_data(loaded) # Sync save
            return loaded
    except (json.JSONDecodeError, IOError) as e:
        print(f"[Server] Error loading data from {DATA_FILE}: {e}. Returning default structure.")
        return default_data
    except Exception as e:
        print(f"[Server] Unexpected error loading data from {DATA_FILE}: {e}. Returning default structure.")
        traceback.print_exc()
        return default_data

def save_data(data: dict[str, Any]):
    """Saves data to the JSON file synchronously."""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"[Server] Error saving data sync to {DATA_FILE}: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"[Server] Unexpected error saving data sync to {DATA_FILE}: {e}")
        traceback.print_exc()

_save_lock = asyncio.Lock()
async def save_data_async(data: dict[str, Any]):
    """Saves data to the JSON file asynchronously and with a lock."""
    async with _save_lock:
        try:
            # Using sync write within the async lock for simplicity
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"[Server] Error saving data async to {DATA_FILE}: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"[Server] Unexpected error saving data async to {DATA_FILE}: {e}")
            traceback.print_exc()

finance_data = load_data()
# --- End Data Persistence ---

mcp = FastMCP("AI_Finance_ERP_Assistant")
print("[Server] Finance Agent MCP Server (Saves Full JSON Conversions) Starting...")

# --- Helpers (find_item_by_id, get_new_id) remain the same ---
# ... (find_item_by_id, get_new_id code remains here) ...
def find_item_by_id(item_type: str, item_id: str) -> Optional[Dict[str, Any]]:
    """Finds an item (transaction, invoice, etc.) by ID."""
    global finance_data
    data_section = finance_data.get(item_type + "s") # e.g., transactions, invoices
    if item_type == "transaction" and isinstance(data_section, list):
        for item in data_section:
            if isinstance(item, dict) and item.get("id") == item_id:
                return item
    elif isinstance(data_section, dict): # e.g., invoices, accounts
        return data_section.get(item_id)
    return None

def get_new_id(prefix: str) -> str:
    """Generates a simple unique ID."""
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


# --- Existing Tools (get_account_balance, ..., query_knowledge_base) remain the same ---
# Make sure all previous tool definitions are included here
# ... (get_account_balance, list_accounts, get_transaction_details, list_transactions, ...)
# ... (add_manual_transaction, categorize_expense, create_invoice, update_invoice_status, ...)
# ... (generate_monthly_expense_report, generate_profit_loss_statement, ...)
# ... (upload_document, list_uploaded_documents, get_document_content, query_knowledge_base ...)
# --- Tool Definitions Go Here ---
@mcp.tool()
async def get_account_balance(account_id: str) -> str:
    """
    Retrieves the current balance for a specific account ID. Read-only.
    Args:
        account_id: The unique identifier for the account (e.g., 'acc-123').
    Returns: JSON string with account details or error.
    """
    print(f"[Server] Request: get_account_balance(account_id={account_id})")
    await asyncio.sleep(0.05) # Simulate minimal delay
    global finance_data
    accounts = finance_data.get("accounts", {})
    account = accounts.get(account_id)
    if account and isinstance(account, dict):
        return json.dumps({"account_id": account_id, "name": account.get('name', 'N/A'), "balance": account.get('balance', 0.0)})
    else:
        return json.dumps({"error": f"Account ID '{account_id}' not found or invalid."})

@mcp.tool()
async def list_accounts() -> str:
    """
    Lists all available account IDs and their names. Read-only.
    Returns: JSON string with a list of accounts or message.
    """
    print(f"[Server] Request: list_accounts()")
    await asyncio.sleep(0.05)
    global finance_data
    accounts = finance_data.get("accounts", {})
    if not accounts or not isinstance(accounts, dict):
         return json.dumps({"message": "No accounts found."})
    accounts_list = [{"account_id": acc_id, "name": details.get("name", "N/A"), "balance": details.get('balance', 0.0)}
                     for acc_id, details in accounts.items() if isinstance(details, dict)]
    return json.dumps(accounts_list)

@mcp.tool()
async def get_transaction_details(transaction_id: str) -> str:
    """
    Retrieves the full details for a specific transaction ID. Read-only.
    Args:
        transaction_id: The unique identifier for the transaction (e.g., 'txn-001').
    Returns: JSON string with transaction details or error.
    """
    print(f"[Server] Request: get_transaction_details(transaction_id={transaction_id})")
    await asyncio.sleep(0.05)
    transaction = find_item_by_id("transaction", transaction_id)
    if transaction:
        return json.dumps(transaction)
    else:
        return json.dumps({"error": f"Transaction ID '{transaction_id}' not found."})

@mcp.tool()
async def list_transactions(account_id: Optional[str] = None, category: Optional[str] = None, month: Optional[int] = None, year: Optional[int] = None, limit: int = 20) -> str:
    """
    Lists transactions, optionally filtering by account, category, month/year. Read-only.
    Args:
        account_id: Optional account ID to filter by. (Note: Ensure transactions have account_id field if using this).
        category: Optional category to filter by (case-insensitive).
        month: Optional month number (1-12) to filter by (requires year).
        year: Optional year number (e.g., 2024) to filter by (requires month).
        limit: Maximum number of transactions to return (default 20).
    Returns: JSON string with a list of matching transactions or message.
    """
    print(f"[Server] Request: list_transactions(account={account_id}, cat={category}, mon={month}, yr={year}, lim={limit})")
    await asyncio.sleep(0.1)
    global finance_data
    transactions: List[Dict[str, Any]] = finance_data.get("transactions", [])
    filtered_txns = []

    if not isinstance(transactions, list):
         return json.dumps({"error": "Internal server error: Transactions data is corrupted."})
    try:
        limit = int(limit)
        if limit <= 0: limit = 20
    except (ValueError, TypeError): limit = 20

    for t in transactions:
        if not isinstance(t, dict): continue
        match = True
        if account_id and t.get('account_id') != account_id: match = False
        if category and t.get('category', '').lower() != category.lower(): match = False
        if month is not None and year is not None:
            try:
                if not (1 <= month <= 12): raise ValueError("Invalid month")
                target_month_str = f"{year}-{month:02d}"
                if not t.get('date', '').startswith(target_month_str): match = False
            except ValueError: return json.dumps({"error": "Invalid month or year for date filtering."})
        elif month is not None or year is not None: return json.dumps({"error": "Both month and year must be provided for date filtering."})
        if match: filtered_txns.append(t)

    limited_txns = sorted(filtered_txns, key=lambda x: x.get('date', ''), reverse=True)[:limit] # Sort by date desc

    if not limited_txns: return json.dumps({"message": "No transactions found matching the criteria."})
    else: return json.dumps(limited_txns)

@mcp.tool()
async def add_manual_transaction(account_id: str, date: str, description: str, amount: float, category: Optional[str] = "Uncategorized") -> str:
    """
    ACTION: Adds a new transaction manually. Requires user confirmation.
    Args:
        account_id: The account ID to associate the transaction with.
        date: The date of the transaction (YYYY-MM-DD format).
        description: A brief description of the transaction.
        amount: The transaction amount (positive for income, negative for expense).
        category: Optional category for the transaction. Defaults to 'Uncategorized'.
    Returns: JSON string confirming the addition with the new transaction ID or error.
    """
    print(f"[Server] ACTION Request: add_manual_transaction(...) account={account_id}, amount={amount}, date={date}, desc={description[:20]}...")
    global finance_data
    if not isinstance(finance_data.get("accounts"), dict) or account_id not in finance_data["accounts"]:
        return json.dumps({"error": f"Account ID '{account_id}' not found."})
    try: datetime.datetime.strptime(date, '%Y-%m-%d')
    except ValueError: return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD."})
    try: amount = float(amount)
    except (ValueError, TypeError): return json.dumps({"error": "Invalid amount. Must be a number."})
    if not description or not isinstance(description, str): return json.dumps({"error": "Invalid description provided."})
    if not category or not isinstance(category, str): category = "Uncategorized"

    new_id = get_new_id("txn")
    new_txn = {
        "id": new_id, "account_id": account_id, "date": date,
        "description": description.strip(), "amount": amount, "category": category.strip()
    }

    if not isinstance(finance_data.get("transactions"), list): finance_data["transactions"] = []
    finance_data["transactions"].append(new_txn)

    new_balance = "Error: Account data invalid"
    account = finance_data["accounts"][account_id]
    if isinstance(account, dict):
        account["balance"] = account.get("balance", 0.0) + amount
        new_balance = account["balance"]
    else: print(f"[Server Warning] Account data for {account_id} became invalid during transaction add.")

    await save_data_async(finance_data)
    print(f"[Server] Added transaction {new_id}")
    return json.dumps({"status": "success", "transaction_id": new_id, "new_balance": new_balance})

@mcp.tool()
async def categorize_expense(transaction_id: str, category: str) -> str:
    """
    ACTION: Assigns or updates the category for a specific transaction ID. Requires user confirmation.
    Args:
        transaction_id: The unique identifier for the transaction (e.g., 'txn-001').
        category: The category to assign (e.g., 'Groceries', 'Dining', 'Travel'). Must be non-empty.
    Returns: JSON string confirming the categorization or error.
    """
    print(f"[Server] ACTION Request: categorize_expense(transaction_id={transaction_id}, category={category})")
    await asyncio.sleep(0.05)
    global finance_data
    if not category or not isinstance(category, str): return json.dumps({"error": "Invalid or empty category provided."})
    transaction = find_item_by_id("transaction", transaction_id)
    if transaction and isinstance(transaction, dict):
        old_category = transaction.get('category', 'N/A')
        transaction['category'] = category.strip()
        await save_data_async(finance_data)
        return json.dumps({"transaction_id": transaction_id, "status": "categorized", "old_category": old_category, "new_category": transaction['category']})
    else: return json.dumps({"error": f"Transaction ID '{transaction_id}' not found or invalid."})

@mcp.tool()
async def create_invoice(customer_name: str, amount: float, due_date: str, description: str) -> str:
    """
    ACTION: Creates a new draft invoice. Requires user confirmation.
    Args:
        customer_name: The name of the customer.
        amount: The total amount of the invoice (must be positive).
        due_date: The date the invoice is due (YYYY-MM-DD).
        description: Description of services/products invoiced.
    Returns: JSON string with the new invoice ID and status or error.
    """
    print(f"[Server] ACTION Request: create_invoice(...) customer={customer_name}, amount={amount}, due={due_date}")
    global finance_data
    if not customer_name or not isinstance(customer_name, str): return json.dumps({"error": "Invalid customer name."})
    if not description or not isinstance(description, str): return json.dumps({"error": "Invalid description."})
    try: datetime.datetime.strptime(due_date, '%Y-%m-%d')
    except ValueError: return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD."})
    try:
        amount = float(amount)
        if amount <= 0: raise ValueError("Amount must be positive")
    except (ValueError, TypeError): return json.dumps({"error": "Invalid amount. Must be positive."})

    new_id = get_new_id("inv")
    new_invoice = {
        "id": new_id, "customer_name": customer_name.strip(), "amount": amount,
        "due_date": due_date, "issue_date": datetime.date.today().isoformat(),
        "description": description.strip(), "status": "draft"
    }
    if not isinstance(finance_data.get("invoices"), dict): finance_data["invoices"] = {}
    finance_data["invoices"][new_id] = new_invoice
    await save_data_async(finance_data)
    print(f"[Server] Created invoice {new_id}")
    return json.dumps({"status": "success", "invoice_id": new_id, "invoice_status": "draft"})

@mcp.tool()
async def update_invoice_status(invoice_id: str, new_status: str) -> str:
    """
    ACTION: Updates the status of an existing invoice (e.g., 'sent', 'paid', 'void'). Requires user confirmation.
    Allowed statuses: draft, sent, paid, overdue, void. Case-insensitive input.
    Args:
        invoice_id: The ID of the invoice to update.
        new_status: The new status to set.
    Returns: JSON string confirming the status update or error.
    """
    print(f"[Server] ACTION Request: update_invoice_status(invoice_id={invoice_id}, new_status={new_status})")
    global finance_data
    allowed_statuses = ["draft", "sent", "paid", "overdue", "void"]
    status_lower = new_status.lower() if isinstance(new_status, str) else ""
    if status_lower not in allowed_statuses: return json.dumps({"error": f"Invalid status '{new_status}'. Allowed: {', '.join(allowed_statuses)}"})

    invoice = find_item_by_id("invoice", invoice_id)
    if invoice and isinstance(invoice, dict):
        old_status = invoice.get('status', 'N/A')
        invoice['status'] = status_lower
        await save_data_async(finance_data)
        print(f"[Server] Updated invoice {invoice_id} status from '{old_status}' to '{invoice['status']}'")
        return json.dumps({"invoice_id": invoice_id, "status": "updated", "old_status": old_status, "new_status": invoice['status']})
    else: return json.dumps({"error": f"Invoice ID '{invoice_id}' not found or invalid."})

@mcp.tool()
async def generate_monthly_expense_report(month: int, year: int) -> str:
    """
    Generates a summary report of expenses for a given month and year. Read-only.
    Args:
        month: The month number (1-12).
        year: The year (e.g., 2024).
    Returns: JSON string containing the expense report summary or message/error.
    """
    print(f"[Server] Request: generate_monthly_expense_report(month={month}, year={year})")
    await asyncio.sleep(0.2)
    global finance_data
    try:
        month = int(month); year = int(year)
        if not (1 <= month <= 12): raise ValueError("Invalid month")
        if not (1900 <= year <= 2100): raise ValueError("Invalid year")
    except (ValueError, TypeError): return json.dumps({"error": "Invalid month or year."})

    target_month_str = f"{year}-{month:02d}"
    monthly_expenses: Dict[str, float] = {}
    total_expenses = 0.0
    transactions: List[Dict[str, Any]] = finance_data.get("transactions", [])
    if not isinstance(transactions, list): return json.dumps({"error": "Internal server error: Transactions data corrupted."})

    for txn in transactions:
        if not isinstance(txn, dict): continue
        txn_date = txn.get('date', ''); txn_amount = txn.get('amount', 0.0)
        is_numeric = isinstance(txn_amount, (int, float))
        if is_numeric and isinstance(txn_date, str) and txn_date.startswith(target_month_str) and txn_amount < 0:
            cat = txn.get('category', 'Uncategorized')
            if not isinstance(cat, str): cat = 'Uncategorized'
            monthly_expenses[cat] = monthly_expenses.get(cat, 0.0) + txn_amount
            total_expenses += txn_amount

    if not monthly_expenses: return json.dumps({"month": month, "year": year, "message": "No expenses found."})
    report = {
        "report_type": "Monthly Expense Summary", "month": month, "year": year,
        "total_expenses": round(total_expenses, 2),
        "expenses_by_category": {cat: round(amount, 2) for cat, amount in monthly_expenses.items()}
    }
    return json.dumps(report, indent=2)

@mcp.tool()
async def generate_profit_loss_statement(start_date: str, end_date: str) -> str:
    """
    Generates a simple Profit and Loss (P&L) statement for a given date range. Read-only.
    Args:
        start_date: The start date of the period (YYYY-MM-DD).
        end_date: The end date of the period (YYYY-MM-DD).
    Returns: JSON string containing the P&L summary or error.
    """
    print(f"[Server] Request: generate_profit_loss_statement(start={start_date}, end={end_date})")
    await asyncio.sleep(0.3)
    global finance_data
    try:
        s_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        e_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        if s_date > e_date: return json.dumps({"error": "Start date cannot be after end date."})
    except (ValueError, TypeError): return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD."})

    total_income, total_expenses = 0.0, 0.0
    income_by_cat, expenses_by_cat = {}, {}
    transactions: List[Dict[str, Any]] = finance_data.get("transactions", [])
    if not isinstance(transactions, list): return json.dumps({"error": "Internal server error: Transactions data corrupted."})

    for txn in transactions:
        if not isinstance(txn, dict): continue
        try:
            txn_date_str = txn.get('date', '')
            if not isinstance(txn_date_str, str): continue
            txn_date = datetime.datetime.strptime(txn_date_str, '%Y-%m-%d').date()
            if s_date <= txn_date <= e_date:
                amount = txn.get('amount', 0.0)
                category = txn.get('category', 'Uncategorized')
                if not isinstance(category, str): category = 'Uncategorized'
                if isinstance(amount, (int, float)):
                    if amount > 0:
                        total_income += amount
                        income_by_cat[category] = income_by_cat.get(category, 0.0) + amount
                    elif amount < 0:
                        total_expenses += amount
                        expenses_by_cat[category] = expenses_by_cat.get(category, 0.0) + amount
        except (ValueError, TypeError): continue # Skip invalid data rows

    net_profit = total_income + total_expenses
    report = {
        "report_type": "Profit and Loss Statement", "period": f"{start_date} to {end_date}",
        "total_income": round(total_income, 2),
        "income_by_category": {c: round(a, 2) for c, a in income_by_cat.items()},
        "total_expenses": round(total_expenses, 2),
        "expenses_by_category": {c: round(a, 2) for c, a in expenses_by_cat.items()},
        "net_profit_or_loss": round(net_profit, 2)
    }
    return json.dumps(report, indent=2)

@mcp.tool()
async def upload_document(file_name: str, file_content: str) -> str:
    """
    ACTION: Uploads the text content of a document to the knowledge base for RAG.
    Use this to add information like ERP documentation, financial policies, account details, etc.
    Args:
        file_name: The desired name for the file (e.g., 'erp_manual_v1.txt'). Must be a safe filename.
        file_content: The full text content of the document.
    Returns: JSON string confirming the upload or error.
    """
    # This tool remains unchanged, it handles simple text uploads.
    # The main file processing is done by `process_uploaded_document`.
    print(f"[Server] ACTION Request: upload_document(file_name='{file_name}')")
    if not isinstance(file_name, str) or not file_name: return json.dumps({"error": "Invalid file name."})
    if not isinstance(file_content, str): return json.dumps({"error": "Invalid file content (must be text)."})
    safe_filename = os.path.basename(file_name)
    if safe_filename != file_name or not safe_filename or '..' in safe_filename or '/' in safe_filename or '\\' in safe_filename:
         return json.dumps({"error": f"Invalid or unsafe file name: '{file_name}'"})
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f: f.write(file_content)
        print(f"[Server] Saved document to '{filepath}'")
        return json.dumps({"status": "success", "message": f"Document '{safe_filename}' added to knowledge base."})
    except IOError as e:
        print(f"[Server] IO Error uploading document '{safe_filename}': {e}")
        traceback.print_exc()
        return json.dumps({"error": f"Failed to save document '{safe_filename}' due to IO Error: {e}"})
    except Exception as e:
        print(f"[Server] Unexpected Error uploading document '{safe_filename}': {e}")
        traceback.print_exc()
        return json.dumps({"error": f"An unexpected error occurred while saving document '{safe_filename}': {e}"})

@mcp.tool()
async def list_uploaded_documents() -> str:
    """
    Lists the names of all documents currently in the knowledge base. Read-only.
    Includes original uploads and generated .json files.
    Returns: JSON string containing a list of filenames or message.
    """
    print(f"[Server] Request: list_uploaded_documents()")
    try:
        if not os.path.exists(UPLOAD_DIR): return json.dumps({"message": "Knowledge base directory not found."})
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        if not files: return json.dumps({"message": "No documents found."})
        else: return json.dumps({"documents": sorted(files)})
    except OSError as e:
        print(f"[Server] OS Error listing documents: {e}")
        traceback.print_exc()
        return json.dumps({"error": f"Error accessing document directory: {e}"})
    except Exception as e:
        print(f"[Server] Unexpected Error listing documents: {e}")
        traceback.print_exc()
        return json.dumps({"error": f"Unexpected error listing documents: {e}"})

@mcp.tool()
async def get_document_content(file_name: str) -> str:
    """
    Retrieves the full text content of a specific document from the knowledge base. Read-only.
    Use this to read original uploads or generated .json files.
    Args:
        file_name: The exact name of the file to retrieve (e.g., 'report.csv' or 'report.csv.json'). Must be a safe filename.
    Returns: JSON string with the file content or an error message.
    """
    print(f"[Server] Request: get_document_content(file_name='{file_name}')")
    if not isinstance(file_name, str) or not file_name: return json.dumps({"error": f"Invalid file name."})
    safe_filename = os.path.basename(file_name)
    if safe_filename != file_name or not safe_filename or '..' in safe_filename or '/' in safe_filename or '\\' in safe_filename:
         return json.dumps({"error": f"Invalid or unsafe file name: '{file_name}'"})
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    if not os.path.isfile(filepath): return json.dumps({"error": f"Document '{safe_filename}' not found."})
    try:
        # Read as text, assuming UTF-8 (most common, including for JSON)
        # Add error handling for potentially large files if needed
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return json.dumps({"file_name": safe_filename, "content": content})
    except IOError as e:
        print(f"[Server] IO Error reading document '{safe_filename}': {e}")
        traceback.print_exc()
        return json.dumps({"error": f"Failed to read document '{safe_filename}' due to IO Error: {e}"})
    except Exception as e:
        print(f"[Server] Unexpected Error reading document '{safe_filename}': {e}")
        traceback.print_exc()
        return json.dumps({"error": f"Unexpected error reading document '{safe_filename}': {e}"})


@mcp.tool()
async def query_knowledge_base(query: str) -> str:
    """
    Searches the uploaded documents (including .json files) for information relevant to the query using basic keyword matching. Read-only.
    Useful for asking questions about procedures, policies, or data found in uploaded documents.
    Args:
        query: The question or keywords to search for (case-insensitive).
    Returns: JSON string with snippets of relevant information or message/error.
    """
    # This tool remains largely the same, but will now naturally search .json files too.
    print(f"[Server] Request: query_knowledge_base(query='{query[:50]}...')")
    if not isinstance(query, str) or not query: return json.dumps({"error": "Invalid or empty query."})
    await asyncio.sleep(0.15)
    relevant_snippets: List[Dict[str, str]] = []
    max_snippets_per_file, total_max_snippets, max_snippet_length = 2, 5, 300
    try:
        if not os.path.exists(UPLOAD_DIR): return json.dumps({"message": "Knowledge base not found."})
        query_lower = query.lower()
        files_searched_count = 0
        for filename in os.listdir(UPLOAD_DIR):
            if len(relevant_snippets) >= total_max_snippets: break
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(filepath):
                files_searched_count += 1
                snippets_from_file = 0
                try:
                    # Read as text, ignoring decode errors for robustness in keyword search
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                    content_lower = content.lower()
                    search_index = 0
                    while snippets_from_file < max_snippets_per_file and len(relevant_snippets) < total_max_snippets:
                        start_index = content_lower.find(query_lower, search_index)
                        if start_index == -1: break
                        # Simple snippet extraction
                        snip_start = max(0, start_index - 50) # Context before
                        snip_end = min(len(content), start_index + len(query) + 250) # Context after
                        snippet_text = content[snip_start:snip_end].strip()
                        prefix = "..." if snip_start > 0 else ""
                        suffix = "..." if snip_end < len(content) else ""
                        relevant_snippets.append({"file": filename, "snippet": f"{prefix}{snippet_text}{suffix}"})
                        snippets_from_file += 1
                        search_index = start_index + 1 # Move past the found instance
                except Exception as read_err:
                    print(f"[Server] Error reading file '{filename}' for RAG query: {read_err}")
        if not relevant_snippets:
            msg = "No documents found." if files_searched_count == 0 else f"No info matching '{query}' found via keyword search ({files_searched_count} docs searched)."
            return json.dumps({"message": msg})
        else:
            result_payload = {"search_type": "basic_keyword", "query": query, "relevant_info": relevant_snippets}
            if len(relevant_snippets) >= total_max_snippets: result_payload["message"] = f"Found {len(relevant_snippets)} relevant snippets (limit reached)."
            return json.dumps(result_payload)
    except Exception as e:
        print(f"[Server] Error querying knowledge base: {e}")
        traceback.print_exc()
        return json.dumps({"error": f"Unexpected error querying knowledge base: {e}"})


# === MODIFIED File Processing Tool ===

@mcp.tool()
async def process_uploaded_document(filename: str, content_base64: str) -> str:
    """
    ACTION: Processes an uploaded file. Saves the original to the knowledge base,
    attempts to convert the full content (CSV, Excel, JSON, TXT) into a structured JSON format,
    and saves this JSON as '<original_filename>.json' in the knowledge base.
    Optionally extracts transactions from CSV/Excel and adds them to finance data.
    Args:
        filename: The original name of the uploaded file.
        content_base64: The base64 encoded content of the file.
    Returns: JSON string summarizing the processing result (original save, JSON creation).
    """
    print(f"[Server] ACTION Request: process_uploaded_document(filename='{filename}')")
    global finance_data # Needed if performing transaction extraction

    # 1. Validate filename and decode content
    if not isinstance(filename, str) or not filename:
        return json.dumps({"status": "error", "message": "Invalid filename provided."})
    if not isinstance(content_base64, str):
        return json.dumps({"status": "error", "message": "Invalid content encoding provided."})

    safe_filename = os.path.basename(filename)
    if safe_filename != filename or not safe_filename or '..' in safe_filename or '/' in safe_filename or '\\' in safe_filename:
        return json.dumps({"status": "error", "message": f"Invalid or unsafe file name provided: '{filename}'"})

    original_filepath = os.path.join(UPLOAD_DIR, safe_filename)
    # Define the name for the generated JSON file
    json_filename = f"{safe_filename}.json"
    json_filepath = os.path.join(UPLOAD_DIR, json_filename)

    summary_messages = []
    saved_original_file = False
    created_json_file = False
    conversion_error = None
    extracted_transaction_count = 0 # Optional: If transaction extraction is enabled
    data_updated = False # Optional: If transaction extraction is enabled

    try:
        content_bytes = base64.b64decode(content_base64)
    except (base64.binascii.Error, ValueError) as decode_err:
        return json.dumps({"status": "error", "message": f"Failed to decode file content: {decode_err}", "filename": safe_filename})

    # 2. Save the original file
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(original_filepath, 'wb') as f:
            f.write(content_bytes)
        saved_original_file = True
        print(f"[Server] Saved original uploaded file to '{original_filepath}'")
        summary_messages.append(f"Original file '{safe_filename}' saved.")
    except IOError as e:
        print(f"[Server] IO Error saving original file '{safe_filename}': {e}")
        # Return error immediately if saving original fails
        return json.dumps({"status": "error", "message": f"IO Error saving original file '{safe_filename}': {e}", "filename": safe_filename})
    except Exception as e:
         print(f"[Server] Unexpected Error saving original file '{safe_filename}': {e}")
         traceback.print_exc()
         return json.dumps({"status": "error", "message": f"Unexpected error saving original file '{safe_filename}': {e}", "filename": safe_filename})

    # 3. Attempt full content conversion to JSON and save as .json file
    file_type = safe_filename.lower().split('.')[-1] if '.' in safe_filename else ''
    full_structured_content = None

    try:
        if file_type == 'csv':
            print(f"[Server] Converting full CSV content from '{safe_filename}' to JSON...")
            content_str = content_bytes.decode('utf-8', errors='replace')
            csvfile = io.StringIO(content_str)
            # Use pandas for robust CSV reading
            df = pd.read_csv(csvfile, low_memory=False) # low_memory=False can help with mixed types
            df = df.where(pd.notnull(df), None) # Replace NaN with None for JSON
            if len(df) > MAX_ROWS_FULL_CONVERSION:
                 print(f"[Server Warning] CSV '{safe_filename}' has {len(df)} rows, limiting JSON conversion to first {MAX_ROWS_FULL_CONVERSION} rows.")
                 df = df.head(MAX_ROWS_FULL_CONVERSION)
                 summary_messages.append(f"Note: CSV content truncated to {MAX_ROWS_FULL_CONVERSION} rows in generated JSON.")
            full_structured_content = {"file_type": "csv", "data": df.to_dict(orient='records')}
            summary_messages.append(f"Converted CSV to JSON structure ({len(df)} rows).")
            # --- Optional: Extract transactions from the dataframe 'df' ---
            # extracted_transaction_count, data_updated = await _extract_transactions_from_dataframe(df, safe_filename)
            # if extracted_transaction_count > 0: summary_messages.append(f"Attempted to extract {extracted_transaction_count} transactions.")
            # --- End Optional ---

        elif file_type in ['xls', 'xlsx']:
            print(f"[Server] Converting full Excel content from '{safe_filename}' to JSON...")
            excel_file = io.BytesIO(content_bytes)
            xls = pd.ExcelFile(excel_file)
            all_sheets_data = {}
            truncated_sheets = []
            for sheet_name in xls.sheet_names:
                 print(f"[Server] Reading sheet: '{sheet_name}'")
                 # Read sheet, try basic header detection
                 try:
                      df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                      if not df.empty and all(isinstance(x, str) for x in df.iloc[0]):
                           df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                      else: # Reread without assuming header
                           df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

                      df = df.where(pd.notnull(df), None) # Replace NaN
                      if len(df) > MAX_ROWS_FULL_CONVERSION:
                          print(f"[Server Warning] Excel sheet '{sheet_name}' in '{safe_filename}' has {len(df)} rows, limiting JSON to first {MAX_ROWS_FULL_CONVERSION} rows.")
                          df = df.head(MAX_ROWS_FULL_CONVERSION)
                          truncated_sheets.append(sheet_name)
                      all_sheets_data[sheet_name] = df.to_dict(orient='records')
                 except Exception as sheet_err:
                      print(f"[Server Error] Could not read sheet '{sheet_name}' from '{safe_filename}': {sheet_err}")
                      all_sheets_data[sheet_name] = {"error": f"Failed to read sheet: {sheet_err}"}

            full_structured_content = {"file_type": "excel", "sheets": all_sheets_data}
            summary_messages.append(f"Converted Excel (all sheets) to JSON structure.")
            if truncated_sheets:
                 summary_messages.append(f"Note: Content truncated to {MAX_ROWS_FULL_CONVERSION} rows in generated JSON for sheets: {', '.join(truncated_sheets)}.")
             # --- Optional: Extract transactions from relevant sheets ---
             # extracted_transaction_count, data_updated = await _extract_transactions_from_excel_sheets(all_sheets_data, safe_filename)
             # if extracted_transaction_count > 0: summary_messages.append(f"Attempted to extract {extracted_transaction_count} transactions.")
             # --- End Optional ---

        elif file_type == 'json':
            print(f"[Server] Validating and saving JSON content from '{safe_filename}'...")
            content_str = content_bytes.decode('utf-8', errors='replace')
            try:
                full_structured_content = json.loads(content_str) # Load to validate
                summary_messages.append("Original file is valid JSON.")
                # We'll save this structure directly to the .json file
            except json.JSONDecodeError as json_err:
                conversion_error = f"Invalid JSON format in uploaded file '{safe_filename}': {json_err}"
                print(f"[Server Error] {conversion_error}")
                summary_messages.append(f"Uploaded file is invalid JSON: {json_err}. Cannot create structured JSON version.")
                # Don't set full_structured_content, so JSON file won't be created

        elif file_type == 'txt':
            print(f"[Server] Wrapping TXT content from '{safe_filename}' in JSON...")
            content_str = content_bytes.decode('utf-8', errors='replace')
            if len(content_str) > MAX_CHARS_FULL_CONVERSION:
                 print(f"[Server Warning] TXT file '{safe_filename}' is large ({len(content_str)} chars), truncating JSON content to {MAX_CHARS_FULL_CONVERSION} chars.")
                 content_str = content_str[:MAX_CHARS_FULL_CONVERSION]
                 summary_messages.append(f"Note: TXT content truncated to {MAX_CHARS_FULL_CONVERSION} characters in generated JSON.")
            full_structured_content = {"file_type": "text", "content": content_str}
            summary_messages.append("Wrapped TXT content in JSON structure.")

        else:
            print(f"[Server] File type '{file_type}' not supported for JSON conversion.")
            summary_messages.append(f"File type '{file_type}' not automatically convertible to JSON. Only original file saved.")
            # No full_structured_content, so no .json file will be created.

        # Save the generated JSON file if conversion was successful
        if full_structured_content is not None:
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(full_structured_content, f, indent=2) # Save the full structure
                created_json_file = True
                print(f"[Server] Saved generated JSON file to '{json_filepath}'")
                summary_messages.append(f"Generated JSON version saved as '{json_filename}'.")
            except IOError as e:
                 conversion_error = f"IO Error saving generated JSON file '{json_filename}': {e}"
                 print(f"[Server Error] {conversion_error}")
                 summary_messages.append(f"Error saving generated JSON: {e}")
                 created_json_file = False # Mark as failed
            except Exception as e:
                 conversion_error = f"Unexpected error saving generated JSON '{json_filename}': {e}"
                 print(f"[Server Error] {conversion_error}")
                 traceback.print_exc()
                 summary_messages.append(f"Unexpected error saving generated JSON: {e}")
                 created_json_file = False # Mark as failed

    except UnicodeDecodeError as ude:
        conversion_error = f"Could not decode file '{safe_filename}' as UTF-8 for conversion: {ude}"
        print(f"[Server Error] {conversion_error}")
        summary_messages.append(f"Decoding error for '{safe_filename}'. Cannot create JSON version.")
    except ImportError as ie:
        # Specifically catch missing optional dependencies like openpyxl
        conversion_error = f"Missing dependency for file type '{file_type}': {ie}. Cannot process."
        print(f"[Server Error] {conversion_error}")
        summary_messages.append(f"Missing library required to process '{safe_filename}': {ie}")
    except Exception as e:
        conversion_error = f"Unexpected error converting file '{safe_filename}' to JSON: {e}"
        print(f"[Server Error] {conversion_error}")
        traceback.print_exc()
        summary_messages.append(f"Unexpected error during JSON conversion: {e}")

    # 4. Return summary
    final_message = " ".join(summary_messages)
    response_status = "error" # Default
    if saved_original_file and created_json_file:
        response_status = "success"
    elif saved_original_file and not created_json_file and full_structured_content is None:
        # Original saved, but type not convertible (expected)
        response_status = "success_original_only"
    elif saved_original_file and not created_json_file and conversion_error:
         # Original saved, but conversion/JSON save failed
         response_status = "partial_failure_conversion"
    elif not saved_original_file:
         # Error should have been returned earlier, but double-check
         response_status = "error"


    response_payload = {
        "status": response_status,
        "message": final_message,
        "filename": safe_filename, # Original filename
        "saved_original": saved_original_file,
        "generated_json_filename": json_filename if created_json_file else None,
        "created_json": created_json_file,
        # Optional fields if transaction extraction is implemented
        # "data_extracted_count": extracted_transaction_count,
        # "data_updated": data_updated
    }
    if conversion_error:
        response_payload["conversion_error"] = conversion_error

    return json.dumps(response_payload)


# Optional Helper for Transaction Extraction from DataFrame (Adapt as needed)
# async def _extract_transactions_from_dataframe(df: pd.DataFrame, source_filename: str) -> tuple[int, bool]:
#    # ... Logic to iterate df.rows, validate, create transaction dicts ...
#    # ... Use _save_lock and save_data_async if modifying finance_data ...
#    print(f"[Server Extract] Processing DataFrame from {source_filename}...")
#    extracted_count = 0
#    data_updated = False
#    # Add extraction logic here, similar to _extract_transactions_from_csv_rows
#    # Remember to handle column name variations robustly
#    return extracted_count, data_updated

# Optional Helper for Transaction Extraction from Excel Sheets Data
# async def _extract_transactions_from_excel_sheets(sheets_data: Dict[str, List[Dict]], source_filename: str) -> tuple[int, bool]:
#     print(f"[Server Extract] Processing Excel sheets from {source_filename}...")
#     total_extracted = 0
#     any_data_updated = False
#     for sheet_name, rows in sheets_data.items():
#         if isinstance(rows, list) and rows: # Check if sheet has valid data
#             # Convert list of dicts to DataFrame for easier processing? Or process directly.
#             # df = pd.DataFrame(rows)
#             # extracted_count, data_updated = await _extract_transactions_from_dataframe(df, f"{source_filename} (Sheet: {sheet_name})")
#             # total_extracted += extracted_count
#             # if data_updated: any_data_updated = True
#             pass # Add extraction logic per sheet here
#     return total_extracted, any_data_updated

# --- Run the Server ---
if __name__ == "__main__":
    print("[Server] Starting Enhanced Finance MCP server (Saves Full JSON Conversions) on stdio...")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"[Server] MCP server run failed: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        print("[Server] MCP server finished.")