"""
Async database wrappers for improved latency with parallel operations.
Uses aiosqlite for async SQLite operations and thread pools for Faiss.
"""

import asyncio
import aiosqlite
from typing import List, Dict, Any, Optional
from pathlib import Path


class AsyncStructuredDatabase:
    
    def __init__(self, db_path: str = "structured_db.sqlite"):
        self.db_path = db_path
    
    async def search(self, client_name: str) -> List[Dict[str, Any]]:
        search_pattern = f"%{client_name}%"
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM clients WHERE client_name LIKE ?",
                (search_pattern,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_all(self) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM clients") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]


class AsyncSemanticDatabase:
    
    def __init__(self, sync_db):
        self.sync_db = sync_db
        self._executor = None
    
    async def search(self, query: str, top_k: int = 7) -> List[Dict[str, Any]]:
        """Async search using thread pool for Faiss operations"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.sync_db.search,
            query,
            top_k
        )


async def parallel_tool_execution(
    structured_query: Optional[str] = None,
    semantic_query: Optional[str] = None,
    structured_db: AsyncStructuredDatabase = None,
    semantic_db: AsyncSemanticDatabase = None
) -> Dict[str, Any]:
    tasks = []
    task_names = []
    
    if structured_query and structured_db:
        tasks.append(structured_db.search(structured_query))
        task_names.append('structured_results')
    
    if semantic_query and semantic_db:
        tasks.append(semantic_db.search(semantic_query))
        task_names.append('semantic_results')
    
    if not tasks:
        return {}
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = {}
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            print(f"[ERROR] {name}: {result}")
            output[name] = []
        else:
            output[name] = result
    
    return output
