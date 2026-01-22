from src.database.structured_db import StructuredDatabase
from src.agents.tools.cache import structured_db_cache, cached_tool
from typing import List, Dict, Any
import asyncio

_structured_db_instance = None

def _get_structured_db():
    global _structured_db_instance
    if _structured_db_instance is None:
        _structured_db_instance = StructuredDatabase()
    return _structured_db_instance


@cached_tool(structured_db_cache, "search_structured_db")
def search_structured_db(client_name: str) -> List[Dict[str, Any]]:
    """
    Search the structured database for the client name.
    Args:
        client_name: The client name to search the structured database for.
    Returns:
        A list of dictionaries containing all client information for matching clients.
    """
    db = _get_structured_db()
    return db.search(client_name)


async def search_structured_db_async(client_name: str) -> List[Dict[str, Any]]:
    """
    Async version of search_structured_db for parallel execution.
    """

    cached_result = structured_db_cache.get("search_structured_db", client_name)
    if cached_result is not None:
        print(f"[CACHE HIT] search_structured_db_async")
        return cached_result
    
    print(f"[CACHE MISS] search_structured_db_async")
    db = _get_structured_db()
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, db.search, client_name)

    structured_db_cache.set("search_structured_db", client_name, result)
    return result
