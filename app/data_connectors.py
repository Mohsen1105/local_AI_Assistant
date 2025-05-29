# app/data_connectors.py
import pandas as pd
from typing import List, Dict, Any

try:
    import pyodbc
except ImportError:
    pyodbc = None
    print("Warning: pyodbc library not found. SQL Server connectivity will be unavailable. Install with: pip install -r requirements-db.txt")

try:
    import cx_Oracle
except ImportError:
    cx_Oracle = None
    print("Warning: cx_Oracle library not found. Oracle DB connectivity will be unavailable. Install with: pip install -r requirements-db.txt")

# --- XLS/XLSX Processing ---
def extract_data_from_xls(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Extracts data from all sheets of an XLS or XLSX file.
    Returns a dictionary where keys are sheet names and values are pandas DataFrames.
    """
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        data_frames = {}
        for sheet_name in sheet_names:
            data_frames[sheet_name] = xls.parse(sheet_name)
        print(f"Successfully extracted {len(sheet_names)} sheets from {file_path}")
        return data_frames
    except Exception as e:
        print(f"Error reading XLS/XLSX file {file_path}: {e}")
        return {} # Return empty dict on error

def dataframe_to_text(df: pd.DataFrame, sheet_name: str) -> str:
    """Converts a DataFrame to a string representation for LLM processing."""
    if df.empty:
        return f"Sheet '{sheet_name}' is empty.\n"
    # Basic text representation, can be made more sophisticated
    text_representation = f"Sheet: {sheet_name}\n"
    text_representation += df.to_string(index=False, na_rep='NaN') + "\n\n"
    return text_representation

# --- SQL Server Connectivity ---
def query_sql_server(connection_string: str, query: str) -> pd.DataFrame:
    """
    Connects to SQL Server, executes a query, and returns results as a DataFrame.
    Example connection_string: 
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=your_server;DATABASE=your_db;UID=your_user;PWD=your_password'
    """
    if pyodbc is None:
        error_message = "pyodbc is not installed. SQL Server functionality is unavailable. Please install dependencies from requirements-db.txt and ensure ODBC drivers are configured."
        print(f"Error: {error_message}")
        raise RuntimeError(error_message)
    try:
        conn = pyodbc.connect(connection_string)
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Successfully executed SQL Server query. Fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error querying SQL Server: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Oracle Database Connectivity ---
def query_oracle_db(dsn: str, user: str, password: str, query: str) -> pd.DataFrame:
    """
    Connects to Oracle DB, executes a query, and returns results as a DataFrame.
    Example dsn (Data Source Name): 'your_host:your_port/your_service_name'
    Ensure Oracle client libraries (like Oracle Instant Client) are set up in the environment.
    """
    if cx_Oracle is None:
        error_message = "cx_Oracle is not installed. Oracle DB functionality is unavailable. Please install dependencies from requirements-db.txt and ensure Oracle Client libraries are configured."
        print(f"Error: {error_message}")
        raise RuntimeError(error_message)
    try:
        # For cx_Oracle.makedsn is often preferred for constructing the DSN string
        # Or use a full connection string if preferred by your setup
        # conn = cx_Oracle.connect(user, password, dsn)
        
        # Simpler connection if Oracle Instant Client is configured with tnsnames.ora or Easy Connect string
        full_connection_string = f"{user}/{password}@{dsn}"
        conn = cx_Oracle.connect(full_connection_string)
        
        cursor = conn.cursor()
        cursor.execute(query)
        # Fetch column names from cursor.description
        columns = [col[0] for col in cursor.description]
        # Fetch all rows
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        
        cursor.close()
        conn.close()
        print(f"Successfully executed Oracle DB query. Fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error querying Oracle DB: {e}")
        # For cx_Oracle, often good to print e.args for more detailed error info from Oracle
        if hasattr(e, 'args') and e.args:
            oracle_error = e.args[0]
            # Check if oracle_error is an instance of cx_Oracle.Error or similar, if cx_Oracle is available
            # For now, we'll keep the hasattr check as it's safer if cx_Oracle might be None
            # though the function should have exited if cx_Oracle was None.
            if hasattr(oracle_error, 'code') and hasattr(oracle_error, 'message'):
                 print(f"Oracle Error Code: {oracle_error.code}")
                 print(f"Oracle Error Message: {oracle_error.message}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Placeholder for Integration into RAG/Summarization ---
# In a real scenario, you would call these functions, convert their output (DataFrames)
# to text (e.g., using dataframe_to_text or more sophisticated serialization),
# and then feed this text into the document_processor's chunking and embedding pipeline
# or directly to an LLM for summarization if the data is small enough.

# Example usage (for testing this module directly):
if __name__ == "__main__":
    # XLS Example (create a dummy test.xlsx for this to run)
    # Sample: Create an excel file 'test.xlsx' with some data
    # dummy_data = {'Sheet1': pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})}
    # with pd.ExcelWriter('test.xlsx') as writer:
    #     for sheet_name, df_sheet in dummy_data.items():
    #         df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
    #
    # if os.path.exists("test.xlsx"):
    #    extracted_excel_data = extract_data_from_xls("test.xlsx")
    #    for sheet, df_sheet in extracted_excel_data.items():
    #        print(dataframe_to_text(df_sheet, sheet))
    # else:
    #    print("test.xlsx not found. Skipping XLS example.")

    # SQL Server Example (requires a running SQL Server and valid connection string)
    # sql_conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=mydatabase;UID=myuser;PWD=mypassword"
    # sql_query_example = "SELECT TOP 5 * FROM MyTable"
    # try:
    #     df_sql = query_sql_server(sql_conn_str, sql_query_example)
    #     if not df_sql.empty:
    #         print("SQL Server Data:")
    #         print(dataframe_to_text(df_sql, "SQL Query Result"))
    #     elif pyodbc.drivers(): # Check if any ODBC drivers are even installed
    #         print("SQL Server query returned empty. Check query and connection. ODBC Drivers available:", pyodbc.drivers())
    #     else:
    #         print("No ODBC drivers found. SQL Server connection will likely fail.")
    # except Exception as e_sql:
    #     print(f"SQL Server example connection failed: {e_sql}")


    # Oracle Example (requires a running Oracle DB, client libs, and valid credentials)
    # oracle_dsn = "localhost:1521/ORCLCDB" # Example DSN
    # oracle_user = "system"
    # oracle_pass = "oracle" # Replace with your actual password
    # oracle_query_example = "SELECT table_name FROM all_tables WHERE ROWNUM <= 5"
    # try:
    #     df_oracle = query_oracle_db(oracle_dsn, oracle_user, oracle_pass, oracle_query_example)
    #     if not df_oracle.empty:
    #         print("Oracle DB Data:")
    #         print(dataframe_to_text(df_oracle, "Oracle Query Result"))
    #     else:
    #         print("Oracle query returned empty. Check DSN, credentials, query, and Oracle client setup.")
    # except Exception as e_oracle:
    #     print(f"Oracle example connection failed: {e_oracle}. Ensure Oracle Instant Client is configured.")
    pass
