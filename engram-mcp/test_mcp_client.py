#!/usr/bin/env python3
"""Test script to verify Engram API and MCP client functionality."""

import asyncio
import sys
import uuid
from engram_mcp.client import EngramClient


async def main():
    # Configuration
    api_url = "http://localhost:8000"
    username = "testuser"
    password = "testpass123"
    conversation_id = str(uuid.uuid4())  # Generate a valid UUID
    
    print(f"Testing Engram API at {api_url}")
    print(f"Username: {username}")
    print("-" * 50)
    
    try:
        # Initialize client
        client = EngramClient(api_url, username, password)
        await client.start()
        print("✓ Client initialized and authenticated")
        
        # Test 1: Health check
        print("\n[Test 1] Health Check")
        health = await client.health_check()
        print(f"  Status: {health.get('status')}")
        print(f"  Version: {health.get('version')}")
        
        # Test 2: Store a memory
        print("\n[Test 2] Store Memory")
        print(f"  Conversation ID: {conversation_id}")
        result = await client.process_turn(
            "I love playing tennis on weekends",
            conversation_id
        )
        print(f"  Operation: {result.get('operation_performed')}")
        print(f"  Memory ID: {result.get('memory_id')}")
        print(f"  Memories affected: {result.get('memories_affected')}")
        
        # Test 3: Store another memory
        print("\n[Test 3] Store Another Memory")
        result2 = await client.process_turn(
            "My favorite programming language is Python",
            conversation_id
        )
        print(f"  Operation: {result2.get('operation_performed')}")
        print(f"  Memory ID: {result2.get('memory_id')}")
        
        # Test 4: Query memories
        print("\n[Test 4] Query Memories")
        query_result = await client.query_memories("What sports do I like?", top_k=5)
        memories = query_result.get("memories", [])
        print(f"  Found {len(memories)} memories")
        for i, mem in enumerate(memories, 1):
            print(f"  {i}. [{mem.get('id')}] {mem.get('text')}")
        
        # Test 5: Direct memory creation
        print("\n[Test 5] Direct Memory Creation")
        direct_result = await client.create_memory(
            "I work as a software engineer",
            importance_score=8.0,
            conversation_id=conversation_id
        )
        print(f"  Created memory ID: {direct_result.get('id')}")
        
        # Test 6: Get statistics
        print("\n[Test 6] Memory Statistics")
        stats = await client.get_stats()
        print(f"  Total memories: {stats.get('total_memories')}")
        print(f"  Average importance: {stats.get('average_importance', 0):.1f}")
        
        # Clean up
        await client.stop()
        print("\n✓ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
