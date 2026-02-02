# Engram API Examples

Quick examples to get started with the Engram API.

## Prerequisites

Make sure the API is running:
```bash
cd engram-backend
docker-compose -f infrastructure/docker/docker-compose.yml up -d
```

API will be available at `http://localhost:8000`

---

## Authentication

### Register a new user
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "securepassword123",
    "full_name": "Test User"
  }'
```

### Login and get token
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=securepassword123"
```

Save the `access_token` from the response for subsequent requests.

---

## Memory Operations

### Process a conversation turn (store memory)
```bash
curl -X POST "http://localhost:8000/memory/process-turn" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "I am vegetarian and I love hiking on weekends",
    "user_id": "user_123",
    "conversation_id": "conv_456"
  }'
```

**Response:**
```json
{
  "turn_id": "abc123",
  "operation_performed": "ADD",
  "memory_id": 1,
  "processing_time_ms": 245.5,
  "memories_affected": 1
}
```

### Query memories
```bash
curl -X POST "http://localhost:8000/memory/query" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are my hobbies?",
    "user_id": "user_123",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "query": "What are my hobbies?",
  "memories": [
    {
      "id": 1,
      "text": "I am vegetarian and I love hiking on weekends",
      "relevance_score": 0.85,
      "timestamp": 1706825600
    }
  ],
  "total_found": 1,
  "processing_time_ms": 89.2
}
```

### Memory consolidation example
```bash
# First memory
curl -X POST "http://localhost:8000/memory/process-turn" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "I completely avoid dairy products",
    "user_id": "user_123",
    "conversation_id": "conv_789"
  }'

# Contradictory update - triggers CONSOLIDATE
curl -X POST "http://localhost:8000/memory/process-turn" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Actually I now eat cheese occasionally",
    "user_id": "user_123",
    "conversation_id": "conv_789"
  }'
```

The system will detect the contradiction and consolidate with temporal awareness:
- "Previously avoided dairy completely, but more recently started eating cheese occasionally"

---

## Health Checks

### Basic health check
```bash
curl http://localhost:8000/health/
```

### Detailed health check
```bash
curl http://localhost:8000/health/detailed
```

---

## Full Workflow Script

```bash
#!/bin/bash
# Full workflow example

BASE_URL="http://localhost:8000"

# 1. Register
echo "Registering user..."
curl -s -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "email": "demo@example.com", "password": "demo123", "full_name": "Demo User"}'

# 2. Login
echo -e "\n\nLogging in..."
TOKEN=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo123" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

echo "Token: $TOKEN"

# 3. Add memories
echo -e "\n\nAdding memories..."
curl -s -X POST "$BASE_URL/memory/process-turn" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I have two dogs named Max and Luna", "user_id": "demo", "conversation_id": "demo_conv"}'

curl -s -X POST "$BASE_URL/memory/process-turn" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I work as a software engineer at a startup", "user_id": "demo", "conversation_id": "demo_conv"}'

# 4. Query
echo -e "\n\nQuerying memories..."
curl -s -X POST "$BASE_URL/memory/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about my pets", "user_id": "demo", "top_k": 5}'

echo -e "\n\nDone!"
```

---

## Python Client Example

```python
import httpx

BASE_URL = "http://localhost:8000"

async def main():
    async with httpx.AsyncClient() as client:
        # Login
        response = await client.post(
            f"{BASE_URL}/auth/login",
            data={"username": "testuser", "password": "securepassword123"}
        )
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Add memory
        response = await client.post(
            f"{BASE_URL}/memory/process-turn",
            headers=headers,
            json={
                "user_message": "I prefer dark mode in all my apps",
                "user_id": "user_123",
                "conversation_id": "conv_001"
            }
        )
        print(f"Added memory: {response.json()}")
        
        # Query
        response = await client.post(
            f"{BASE_URL}/memory/query",
            headers=headers,
            json={
                "query": "What are my UI preferences?",
                "user_id": "user_123",
                "top_k": 5
            }
        )
        print(f"Query result: {response.json()}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
