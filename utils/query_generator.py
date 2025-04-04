# pip install fastapi uvicorn python-dotenv openai graphql-core requests

from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from openai import OpenAI
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API settings
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT", "https://squid.subsquid.io/polkadot-polkassembly/graphql")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

config = Config()

# Initialize OpenAI client
client = OpenAI(api_key=config.OPENAI_API_KEY)

# Create FastAPI app
app = FastAPI(
    title="Polkadot Governance Natural Language API",
    description="Query Polkadot governance data using natural language"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class NLQueryRequest(BaseModel):
    query: str
    include_schema_context: bool = True
    include_query_in_response: bool = True

# Response models
class GraphQLResponse(BaseModel):
    data: Dict[str, Any]
    graphql_query: Optional[str] = None
    explanation: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    graphql_query: Optional[str] = None
    suggestion: Optional[str] = None

# Schema context for LLM
SCHEMA_CONTEXT = """
You are translating natural language queries into GraphQL for a Polkadot governance data API.
The schema contains the following main types:

1. Proposal - Central entity representing various governance proposals (referendums, bounties, etc.)
   - Key fields: id, type, description, proposer, status, createdAt, endedAt
   - Related entities: votes, convictionVoting, statusHistory

2. Vote - Records votes on proposals
   - Key fields: voter, decision (yes/no/abstain), balance
   - Related to: Proposal

3. ConvictionVote - Conviction voting system specific to Polkadot
   - Key fields: voter, lockPeriod, selfVotingPower, delegatedVotingPower
   - Related to: Proposal

4. VotingDelegation - Records when users delegate their votes to others
   - Key fields: from, to, balance, lockPeriod

5. Preimage - Contains proposal details and call data
   - Key fields: hash, proposer, section, method

6. StatusHistory - Tracks status changes of proposals
   - Key fields: status, timestamp, block

7. ProposalGroup - Groups related proposals
   - Helps connect proposals across different governance processes

Filtering is done using the _where argument with operators like _eq, _gt, _lt, etc.
For example: where: { status_eq: PASSED, createdAt_gte: "2023-01-01" }

Ordering uses the orderBy argument, for example: orderBy: [createdAt_DESC]

Pagination uses limit and offset parameters.
"""

# Helper function to generate GraphQL query from natural language
def generate_graphql_query(natural_language_query: str, include_schema_context: bool = True) -> Dict[str, str]:
    # Construct the prompt with schema information if provided
    if include_schema_context:
        prompt = f"""
        {SCHEMA_CONTEXT}
        
        Convert this natural language query to a valid GraphQL query:
        "{natural_language_query}"
        
        First think about what the user is asking for, then construct a precise GraphQL query.
        Include only the fields that would be relevant to answer the user's question.
        Return JSON with the following structure:
        {{
          "graphql_query": "the full GraphQL query",
          "explanation": "brief explanation of what the query does"
        }}
        """
    else:
        prompt = f"""
        Convert this natural language query to a valid GraphQL query for the Polkadot governance API:
        "{natural_language_query}"
        
        Return JSON with the following structure:
        {{
          "graphql_query": "the full GraphQL query",
          "explanation": "brief explanation of what the query does"
        }}
        """
    
    # Call the language model
    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a specialized AI that converts natural language to GraphQL queries for Polkadot governance data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1  # Low temperature for consistent outputs
    )
    
    # Extract the generated JSON response
    try:
        generated_text = response.choices[0].message.content
        
        # Try to parse the response as JSON
        try:
            result = json.loads(generated_text)
            
            # Validate that the result has the expected structure
            if not isinstance(result, dict) or "graphql_query" not in result:
                # If the result doesn't have the expected structure, try to extract a GraphQL query
                if isinstance(result, dict) and any(key for key in result.keys() if "query" in key.lower()):
                    # Find a key that might contain the query
                    query_key = next(key for key in result.keys() if "query" in key.lower())
                    result = {
                        "graphql_query": result[query_key],
                        "explanation": "Extracted query from response"
                    }
                else:
                    # If we can't find a query, use the entire response as the query
                    result = {
                        "graphql_query": generated_text,
                        "explanation": "Using raw response as query"
                    }
        except json.JSONDecodeError:
            # If the response is not valid JSON, try to extract JSON from the text
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json.loads(json_match.group(0))
                    if isinstance(extracted_json, dict) and "graphql_query" in extracted_json:
                        result = extracted_json
                    else:
                        # Try to find a field that might contain the query
                        query_field = next((k for k in extracted_json.keys() if "query" in k.lower()), None)
                        if query_field:
                            result = {
                                "graphql_query": extracted_json[query_field],
                                "explanation": "Extracted query from JSON"
                            }
                        else:
                            result = {
                                "graphql_query": generated_text,
                                "explanation": "Could not find query in JSON. Using raw response."
                            }
                except (json.JSONDecodeError, StopIteration):
                    # If still not valid JSON or no query field found, create a structured response
                    result = {
                        "graphql_query": generated_text,
                        "explanation": "Could not parse response as JSON. Using raw response as query."
                    }
            else:
                # If no JSON found, check if the text looks like a GraphQL query
                if "{" in generated_text and "}" in generated_text:
                    result = {
                        "graphql_query": generated_text,
                        "explanation": "Using raw response as GraphQL query"
                    }
                else:
                    # If it doesn't look like a GraphQL query, create a structured response
                    result = {
                        "graphql_query": "",
                        "explanation": "Could not extract a valid GraphQL query from the response."
                    }
        return result
    except Exception as e:
        return {
            "graphql_query": "",
            "explanation": f"Error generating query: {str(e)}",
            "error": str(e)
        }

# Execute GraphQL query
def execute_graphql_query(query: str) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
    }
    
    # Ensure we're sending a valid GraphQL query string
    if not query or not isinstance(query, str):
        return {"error": "Invalid GraphQL query: Query must be a non-empty string"}
    
    # Print the query for debugging
    print("\n=== GENERATED GRAPHQL QUERY ===")
    print(query)
    print("==============================\n")
    
    try:
        response = requests.post(
            config.GRAPHQL_ENDPOINT,
            headers=headers,
            json={"query": query}
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"GraphQL query execution failed: {response.text}"
            )
        
        result = response.json()
        
        # Check for GraphQL errors
        if "errors" in result:
            error_messages = "; ".join([error.get("message", "Unknown error") for error in result["errors"]])
            return {
                "error": f"GraphQL errors: {error_messages}",
                "data": result.get("data", {})
            }
        
        return {"data": result.get("data", {})}
    
    except requests.RequestException as e:
        return {"error": f"Request error: {str(e)}"}

# API endpoints
@app.post("/query", response_model=GraphQLResponse)
async def process_natural_language_query(request: NLQueryRequest):
    # Generate GraphQL query from natural language
    generated_result = generate_graphql_query(
        request.query, 
        include_schema_context=request.include_schema_context
    )
    
    if "error" in generated_result:
        raise HTTPException(status_code=400, detail=generated_result["error"])
    
    # Extract the actual GraphQL query string
    graphql_query = generated_result.get("graphql_query", "")
    explanation = generated_result.get("explanation", "")
    
    # Print the explanation for debugging
    print("\n=== QUERY EXPLANATION ===")
    print(explanation)
    print("========================\n")
    
    if not graphql_query:
        raise HTTPException(status_code=400, detail="Failed to generate GraphQL query")
    
    # Execute the query against the GraphQL endpoint
    result = execute_graphql_query(graphql_query)
    
    if "error" in result:
        # If there's an error, return it with the generated query for debugging
        raise HTTPException(
            status_code=400, 
            detail=ErrorResponse(
                error=result["error"],
                graphql_query=graphql_query if request.include_query_in_response else None,
                suggestion="Try rephrasing your query or being more specific"
            ).dict()
        )
    
    # Return both the data and optionally the generated query
    return GraphQLResponse(
        data=result["data"],
        graphql_query=graphql_query if request.include_query_in_response else None,
        explanation=explanation
    )

# Example queries endpoint - provide sample queries to help users
@app.get("/example-queries")
async def get_example_queries():
    return {
        "examples": [
            {
                "natural_language": "Show me the latest 5 referendums",
                "graphql_query": """
                {
                  proposals(
                    where: {type_eq: ReferendumV2}
                    orderBy: [createdAt_DESC]
                    limit: 5
                  ) {
                    id
                    description
                    status
                    createdAt
                    endedAt
                  }
                }
                """
            },
            {
                "natural_language": "Find proposals with more than 100 votes",
                "graphql_query": """
                {
                  proposals(
                    where: {
                      voting_some: {}
                    }
                  ) {
                    id
                    type
                    description
                    status
                    createdAt
                    voting(limit: 1) {
                      id
                    }
                    _count {
                      voting
                    }
                  }
                }
                """
            },
            {
                "natural_language": "Who are the top 10 most active voters?",
                "graphql_query": """
                {
                  flattenedConvictionVotesConnection(
                    orderBy: [voter_ASC]
                  ) {
                    edges {
                      node {
                        voter
                        _count {
                          votes
                        }
                      }
                    }
                  }
                }
                """
            }
        ]
    }

# Schema info endpoint - provides information about the schema
@app.get("/schema-info")
async def get_schema_info():
    return {
        "main_entities": [
            {
                "name": "Proposal",
                "description": "Central entity for all governance proposals",
                "key_fields": ["id", "type", "description", "status", "createdAt"]
            },
            {
                "name": "Vote",
                "description": "Records votes on proposals",
                "key_fields": ["voter", "decision", "balance"]
            },
            {
                "name": "ConvictionVote",
                "description": "Conviction voting records with vote locking",
                "key_fields": ["voter", "lockPeriod", "totalVotingPower"]
            },
            {
                "name": "VotingDelegation",
                "description": "Tracks delegated voting",
                "key_fields": ["from", "to", "balance", "lockPeriod"]
            }
        ],
        "query_capabilities": {
            "filtering": "Use _where with operators like _eq, _gt, _lt",
            "ordering": "Use orderBy with field_ASC or field_DESC",
            "pagination": "Use limit and offset parameters",
            "connections": "Use entity+'Connection' (e.g., proposalsConnection) for cursor-based pagination"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Debug endpoint - only active when DEBUG_MODE is true
@app.post("/debug/query")
async def debug_query(request: NLQueryRequest, response: Response):
    if not config.DEBUG_MODE:
        response.status_code = 404
        return {"error": "Debug mode is disabled"}
    
    # Generate query without executing it
    generated_result = generate_graphql_query(
        request.query, 
        include_schema_context=request.include_schema_context
    )
    
    return {
        "natural_language_query": request.query,
        "graphql_query": generated_result.get("graphql_query", ""),
        "explanation": generated_result.get("explanation", ""),
        "schema_context_included": request.include_schema_context
    }

# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)