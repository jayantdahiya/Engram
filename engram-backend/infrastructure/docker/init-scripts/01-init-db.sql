-- Initialize Engram Database
-- This script sets up the database schema for the Engram system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(100),
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create conversation_turns table
CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    assistant_response TEXT,
    turn_number INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    memory_operations TEXT[] DEFAULT '{}',
    processing_time_ms FLOAT DEFAULT 0.0
);

-- Create memories table with vector support
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    text TEXT NOT NULL,
    embedding VECTOR(1536), -- OpenAI ada-002 dimension
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    importance_score FLOAT DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_conversation_id ON memories(conversation_id);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance_score);
CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);

-- Create vector similarity index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_memories_embedding_cosine 
ON memories USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create HNSW index for better vector search performance (if supported)
-- CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
-- ON memories USING hnsw (embedding vector_cosine_ops);

-- Create indexes for conversation tables
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation_id ON conversation_turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_user_id ON conversation_turns(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_timestamp ON conversation_turns(timestamp);

-- Create indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at 
    BEFORE UPDATE ON conversations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memories_updated_at 
    BEFORE UPDATE ON memories 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to calculate memory importance based on access count
CREATE OR REPLACE FUNCTION calculate_importance_score(access_count INTEGER)
RETURNS FLOAT AS $$
BEGIN
    RETURN LN(1 + access_count);
END;
$$ LANGUAGE plpgsql;

-- Create function to update importance score when access count changes
CREATE OR REPLACE FUNCTION update_importance_score()
RETURNS TRIGGER AS $$
BEGIN
    NEW.importance_score = calculate_importance_score(NEW.access_count);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update importance score
CREATE TRIGGER update_memories_importance 
    BEFORE UPDATE ON memories 
    FOR EACH ROW 
    WHEN (OLD.access_count IS DISTINCT FROM NEW.access_count)
    EXECUTE FUNCTION update_importance_score();

-- Create view for memory statistics
CREATE OR REPLACE VIEW memory_stats AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(m.id) as total_memories,
    AVG(m.importance_score) as avg_importance,
    MAX(m.timestamp) as last_memory_created,
    SUM(m.access_count) as total_accesses
FROM users u
LEFT JOIN memories m ON u.id = m.user_id
GROUP BY u.id, u.username;

-- Create view for conversation statistics
CREATE OR REPLACE VIEW conversation_stats AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(c.id) as total_conversations,
    COUNT(ct.id) as total_turns,
    AVG(conversation_turn_counts.turn_count) as avg_turns_per_conversation,
    MAX(c.updated_at) as last_conversation_activity
FROM users u
LEFT JOIN conversations c ON u.id = c.user_id
LEFT JOIN conversation_turns ct ON c.id = ct.conversation_id
LEFT JOIN (
    SELECT conversation_id, COUNT(*) as turn_count
    FROM conversation_turns
    GROUP BY conversation_id
) conversation_turn_counts ON c.id = conversation_turn_counts.conversation_id
GROUP BY u.id, u.username;

-- Insert default admin user (password: admin123 - change in production!)
INSERT INTO users (id, username, email, full_name, hashed_password, is_active)
VALUES (
    uuid_generate_v4(),
    'admin',
    'admin@mem0.local',
    'System Administrator',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8Kz8KzK', -- admin123
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO engram_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO engram_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO engram_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO engram_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO engram_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO engram_user;
