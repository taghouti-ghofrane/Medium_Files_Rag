"""
Database query tool
"""
import logging
import sqlite3
import json
from typing import Dict, Any, List, Callable, Optional, Tuple
from langchain_community.utilities import SQLDatabase




# Configure logging
logger = logging.getLogger(__name__)

class DatabaseService:
    """
    SQLite-based database query service
    """
    
    # 1. Initialize database query service
    def __init__(self, db_path: str):
        """
        Initialize database query service
        
        Args:
            db_path: Database file path
        """
        self.db_path = db_path
        logger.info("Database query service initialized successfully")
    
    # 2. Query database table structure
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Query structure of specified table
        
        Args:
            table_name: Table name
            
        Returns:
            Dict[str, Any]: Table structure information
        """
        result = {
            "status": "error",
            "data": None,
            "message": ""
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query table structure
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            if columns:
                result["status"] = "success"
                result["data"] = [{"name": col[1], "type": col[2]} for col in columns]
            else:
                result["message"] = f"Table {table_name} does not exist or has no column information"
            
            conn.close()
        except Exception as e:
            logger.error(f"Error occurred when querying table structure: {str(e)}")
            result["message"] = f"Error occurred when querying table structure: {str(e)}"
        
        return result
    
    # 3. Query database data
    def query_data(self, table_name: str, query: str) -> Dict[str, Any]:
        """
        Query data from specified table
        
        Args:
            table_name: Table name
            query: Query condition (SQL statement fragment)
            
        Returns:
            Dict[str, Any]: Query result
        """
        result = {
            "status": "error",
            "data": None,
            "message": ""
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Construct query statement
            full_query = f"SELECT * FROM {table_name} WHERE {query}"
            cursor.execute(full_query)
            rows = cursor.fetchall()
            
            if rows:
                result["status"] = "success"
                result["data"] = rows
            else:
                result["message"] = f"No data found matching criteria"
            
            conn.close()
        except Exception as e:
            logger.error(f"Error occurred when querying data: {str(e)}")
            result["message"] = f"Error occurred when querying data: {str(e)}"
        
        return result

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

class DatabaseTools:
    """
    Database query tool
    """
    
    def __init__(self, db_path: str):
        """
        Initialize database tool
        
        Args:
            db_path: Database file path
        """
        self.db_service = DatabaseService(db_path)
        self.db = SQLDatabase.from_uri('sqlite:///D:\\adavance\\bigmodel\\2.原创案例：Agentic RAG智能问答系统Agent\\chinook.db')
        self.generate_query_system_prompt = """
        You are an agent designed to interact with SQL databases.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the query results and return the answer. Unless the user explicitly specifies they want a certain number of examples,
        always limit the query to at most {top_k} results.

        You can order the results by relevant columns to return the most interesting examples from the database.
        Never query all columns from a specific table, only ask for columns relevant to the question.

        Do not execute any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) on the database.
        """.format(
            dialect=self.db.dialect,
            top_k=5,
        )

        self.query_check_system = """You are a detail-oriented SQL expert.
        Please carefully check for common errors in SQLite queries, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If any of the above errors are found, please rewrite the query. If there are no errors, return the query statement as is.

        After completing the check, you will call the appropriate tool to execute the query."""

        self.check_query_system_prompt = """
        You are a SQL expert with high attention to detail.
        Carefully check for common errors in {dialect} queries, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If any of the above errors exist, please rewrite the query. If there are no errors,
        simply reproduce the original query.

        After running this check, you will call the appropriate tool to execute the query.
        """.format(dialect=self.db.dialect)

        logger.info("Database query tool initialized successfully")
    
    def query_table_schema(self, table_name: str) -> str:
        """
        Query structure of specified table
        
        Args:
            table_name: Table name
            
        Returns:
            str: Table structure information
        """
        result = self.db_service.get_table_schema(table_name)
        if result["status"] == "success":
            schema_info = json.dumps(result["data"], ensure_ascii=False, indent=2)
            return f"Table {table_name} structure information:\n{schema_info}"
        else:
            return f"Failed to get structure of table {table_name}: {result.get('message', 'Unknown error')}"
    
    def query_table_data(self, table_name: str, query: str) -> str:
        """
        Query data from specified table
        
        Args:
            table_name: Table name
            query: Query condition (SQL statement fragment)
            
        Returns:
            str: Query result
        """
        result = self.db_service.query_data(table_name, query)
        if result["status"] == "success":
            data_info = json.dumps(result["data"], ensure_ascii=False, indent=2)
            return f"Query result:\n{data_info}"
        else:
            return f"Query data failed: {result.get('message', 'Unknown error')}"
    
    def list_tables(self) -> str:
        """Input is an empty string, returns all tables in the database: comma-separated list of table names"""
        return ", ".join(self.db.get_usable_table_names())  #   ['emp': "This is an employee table,", '']
        
    def query_text2sql(self, query: str) -> str:
        """
        Execute SQL query and return result.
        If query is incorrect, returns error message.
        If error is returned, please rewrite the query statement, check and retry.

        Args:
            query (str): SQL query statement to execute

        Returns:
            str: Query result or error message
        """
        result = self.db.run_no_throw(query)  # Execute query (does not throw exception)
        if not result:
            return "Error: Query failed. Please modify the query statement and retry."
        return result
        

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DATABASE_PATH = r"D:\adavance\bigmodel\2.原创案例：Agentic RAG智能问答系统Agent\chinook.db"

# Initialize database tool
db_tools = DatabaseTools(db_path=DATABASE_PATH)
print(db_tools.list_tables())

# Test query table structure
def test_query_table_schema():
    table_name = "artists"  # Assume this table exists in the database
    result = db_tools.query_table_schema(table_name)
    print(f"Query table structure result:\n{result}")

# Test query table data
def test_query_table_data():
    table_name = "test_1"  # Assume this table exists in the database
    query = "department = 'Sales'"  # Assume table has department field
    result = db_tools.query_table_data(table_name, query)
    print(f"Query table data result:\n{result}")

# Main function
if __name__ == "__main__":
    logger.info("Starting to test database query tool")
    
    # Test query table structure
    test_query_table_schema()
    
    # Test query table data
    test_query_table_data()
    
    logger.info("Test completed")