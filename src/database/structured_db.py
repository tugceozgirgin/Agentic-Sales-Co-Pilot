import sqlite3
import json
import os
from typing import List, Dict, Any
from pathlib import Path
from src.database.base_database import BaseDatabase


class StructuredDatabase(BaseDatabase):
    """
    SQLite-based structured database implementation.
    Stores and manages client information in a relational database.
    """
    
    def __init__(self, db_path: str = "structured_db.sqlite"):
        super().__init__()
        self.db_path = db_path
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row 
        return conn
    
    def _init_database(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                client_name TEXT NOT NULL,
                industry TEXT,
                total_spend_ytd REAL,
                last_meeting_date TEXT,
                account_manager TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def search(self, client_name: str) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{client_name}%"
        
        cursor.execute("""
            SELECT * FROM clients
            WHERE client_name LIKE ?
        """, (search_pattern,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        print(f"Results: {results}")
        return results
    
    
    def populate(self, data: List[Dict[str, Any]]) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for record in data:
                cursor.execute("""
                    INSERT OR REPLACE INTO clients 
                    (client_id, client_name, industry, total_spend_ytd, last_meeting_date, account_manager)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    record.get('client_id'),
                    record.get('client_name'),
                    record.get('industry'),
                    record.get('total_spend_ytd'),
                    record.get('last_meeting_date'),
                    record.get('account_manager')
                ))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"Error populating database: {e}")
            return False
    
    def populate_from_json(self, json_path: str = "mock_sql_data.json") -> bool:
        try:
            current_dir = Path(__file__).parent.parent
            json_file = current_dir / json_path
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.populate(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}")
            return False
    
    def get_all(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM clients")
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    @staticmethod
    def initialize_and_populate(db_path: str = "structured_db.sqlite",
                                json_path: str = "mock_sql_data.json"):

        print("[INFO] Initializing structured database...")
        try:
            db = StructuredDatabase(db_path=db_path)
            success = db.populate_from_json(json_path=json_path)
            if success:
                print("[INFO] Structured database initialized and populated successfully.")
            else:
                print("[WARNING] Structured database population returned False.")
            return db
        except Exception as e:
            print(f"[ERROR] Failed to initialize structured database: {e}")
            raise

# if __name__ == "__main__":
#     db = StructuredDatabase(db_path="structured_db.sqlite")
#     db.populate_from_json()
#     results = db.search(120000)
#     print(results)
