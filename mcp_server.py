from fastmcp import FastMCP, Context

mcp = FastMCP("Web Search MCP Server")

@mcp.tool
async def search(query: str, ctx: Context):
    
    await ctx.info(f"Searching for {query} on Web...")

    return["some results", "results2", "results3"]

mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")