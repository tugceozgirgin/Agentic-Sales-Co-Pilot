from src.database.semantic_db import SemanticDatabase
from src.agents.tools.cache import semantic_db_cache, cached_tool
from typing import List, Dict, Any
import asyncio

# Create a singleton instance to avoid reinitializing the database multiple times
_semantic_db_instance = None

def _get_semantic_db():
    global _semantic_db_instance
    if _semantic_db_instance is None:
        _semantic_db_instance = SemanticDatabase()
    return _semantic_db_instance


@cached_tool(semantic_db_cache, "search_semantic_db")
def search_semantic_db(query: str) -> List[Dict[str, Any]]:
    """
    Search the semantic Faiss database for retrieving sales rules and policies.
    Args:
        query: The query to search the semantic database for.
    Returns:
        A list of dictionaries containing the sales rules and policies for the query. 
        The dictionary contains the text, distance, and index of the retrieved sales rules and policies.
    """
    db = _get_semantic_db()
    return db.search(query, top_k=7)


async def search_semantic_db_async(query: str) -> List[Dict[str, Any]]:
    """
    Async version of search_semantic_db for parallel execution.
    """
    cached_result = semantic_db_cache.get("search_semantic_db", query)
    if cached_result is not None:
        print(f"[CACHE HIT] search_semantic_db_async")
        return cached_result
    
    print(f"[CACHE MISS] search_semantic_db_async")
    db = _get_semantic_db()
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, db.search, query, 7)
    
    semantic_db_cache.set("search_semantic_db", query, result)
    return result
