"""Graph service for Neo4j operations (Engram Graph)"""

import time
from typing import Any

from neo4j import AsyncGraphDatabase

from core.config import settings
from core.logging import logger


class GraphService:
    """Service for managing graph-based memory relationships (Engram Graph)"""

    def __init__(self):
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.driver: AsyncGraphDatabase | None = None

    async def initialize(self):
        """Initialize Neo4j driver"""
        try:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            await self.driver.verify_connectivity()
            logger.info("Neo4j driver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    async def close(self):
        """Close Neo4j driver"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j driver closed")

    async def store_graph_memory(
        self,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        user_id: str,
        session,
    ):
        """Store entities and relationships in Neo4j"""
        try:
            timestamp = time.time()

            # Create entities
            for entity in entities:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id, user_id: $user_id})
                    SET e.name = $name,
                        e.type = $type,
                        e.confidence = $confidence,
                        e.timestamp = $timestamp,
                        e.last_updated = $timestamp
                    """,
                    id=entity["id"],
                    user_id=user_id,
                    name=entity["name"],
                    type=entity["type"],
                    confidence=entity.get("confidence", 1.0),
                    timestamp=timestamp,
                )

            # Create relationships
            for rel in relationships:
                await session.run(
                    """
                    MATCH (s:Entity {id: $source, user_id: $user_id})
                    MATCH (t:Entity {id: $target, user_id: $user_id})
                    MERGE (s)-[r:RELATION {type: $relation, user_id: $user_id}]->(t)
                    SET r.confidence = $confidence,
                        r.timestamp = $timestamp,
                        r.last_updated = $timestamp
                    """,
                    source=rel["source"],
                    target=rel["target"],
                    relation=rel["relation"],
                    user_id=user_id,
                    confidence=rel.get("confidence", 1.0),
                    timestamp=timestamp,
                )

            logger.debug(
                f"Stored {len(entities)} entities and {len(relationships)} relationships for user {user_id}"
            )

        except Exception as e:
            logger.error(f"Failed to store graph memory: {e}")
            raise

    async def get_entity_relationships(
        self, entity_id: str, user_id: str, session, max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """Get relationships for a specific entity"""
        try:
            result = await session.run(
                """
                MATCH path = (e:Entity {id: $entity_id, user_id: $user_id})-[*1..$max_depth]-(connected)
                RETURN path, length(path) as depth
                ORDER BY depth
                """,
                entity_id=entity_id,
                user_id=user_id,
                max_depth=max_depth,
            )

            relationships = []
            async for record in result:
                path = record["path"]
                depth = record["depth"]

                # Extract relationship information
                rel_info = {"depth": depth, "nodes": [], "relationships": []}

                for node in path.nodes:
                    rel_info["nodes"].append(
                        {"id": node.get("id"), "name": node.get("name"), "type": node.get("type")}
                    )

                for rel in path.relationships:
                    rel_info["relationships"].append(
                        {"type": rel.type, "confidence": rel.get("confidence", 1.0)}
                    )

                relationships.append(rel_info)

            return relationships

        except Exception as e:
            logger.error(f"Failed to get entity relationships: {e}")
            return []

    async def find_related_entities(
        self,
        query_entities: list[str],
        user_id: str,
        session,
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find entities related to the query entities"""
        try:
            if relationship_types:
                rel_filter = f"AND type(r) IN {relationship_types}"
            else:
                rel_filter = ""

            result = await session.run(
                f"""
                MATCH (e:Entity {{user_id: $user_id}})-[r:RELATION{rel_filter}]-(related:Entity {{user_id: $user_id}})
                WHERE e.id IN $query_entities
                RETURN DISTINCT related.id as id, 
                       related.name as name, 
                       related.type as type,
                       collect(DISTINCT type(r)) as relationship_types,
                       collect(DISTINCT r.confidence) as confidences
                ORDER BY size(relationship_types) DESC
                """,
                query_entities=query_entities,
                user_id=user_id,
            )

            related_entities = []
            async for record in result:
                related_entities.append(
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "relationship_types": record["relationship_types"],
                        "avg_confidence": sum(record["confidences"]) / len(record["confidences"]),
                    }
                )

            return related_entities

        except Exception as e:
            logger.error(f"Failed to find related entities: {e}")
            return []

    async def get_user_entity_graph(
        self, user_id: str, session, limit: int = 100
    ) -> dict[str, Any]:
        """Get complete entity graph for a user"""
        try:
            # Get all entities
            entities_result = await session.run(
                """
                MATCH (e:Entity {user_id: $user_id})
                RETURN e.id as id, e.name as name, e.type as type, e.timestamp as timestamp
                ORDER BY e.timestamp DESC
                LIMIT $limit
                """,
                user_id=user_id,
                limit=limit,
            )

            entities = []
            async for record in entities_result:
                entities.append(
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "timestamp": record["timestamp"],
                    }
                )

            # Get all relationships
            relationships_result = await session.run(
                """
                MATCH (s:Entity {user_id: $user_id})-[r:RELATION]->(t:Entity {user_id: $user_id})
                RETURN s.id as source, t.id as target, type(r) as relation, r.confidence as confidence
                """,
                user_id=user_id,
            )

            relationships = []
            async for record in relationships_result:
                relationships.append(
                    {
                        "source": record["source"],
                        "target": record["target"],
                        "relation": record["relation"],
                        "confidence": record["confidence"],
                    }
                )

            return {
                "entities": entities,
                "relationships": relationships,
                "total_entities": len(entities),
                "total_relationships": len(relationships),
            }

        except Exception as e:
            logger.error(f"Failed to get user entity graph: {e}")
            return {
                "entities": [],
                "relationships": [],
                "total_entities": 0,
                "total_relationships": 0,
            }

    async def search_entities_by_name(
        self, name_pattern: str, user_id: str, session, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search entities by name pattern"""
        try:
            result = await session.run(
                """
                MATCH (e:Entity {user_id: $user_id})
                WHERE toLower(e.name) CONTAINS toLower($pattern)
                RETURN e.id as id, e.name as name, e.type as type, e.confidence as confidence
                ORDER BY e.confidence DESC, e.timestamp DESC
                LIMIT $limit
                """,
                pattern=name_pattern,
                user_id=user_id,
                limit=limit,
            )

            entities = []
            async for record in result:
                entities.append(
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "confidence": record["confidence"],
                    }
                )

            return entities

        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []

    async def get_entity_statistics(self, user_id: str, session) -> dict[str, Any]:
        """Get statistics about user's entity graph"""
        try:
            # Entity type distribution
            type_result = await session.run(
                """
                MATCH (e:Entity {user_id: $user_id})
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
                """,
                user_id=user_id,
            )

            entity_types = {}
            async for record in type_result:
                entity_types[record["type"]] = record["count"]

            # Relationship type distribution
            rel_result = await session.run(
                """
                MATCH ()-[r:RELATION {user_id: $user_id}]->()
                RETURN type(r) as relation_type, count(r) as count
                ORDER BY count DESC
                """,
                user_id=user_id,
            )

            relationship_types = {}
            async for record in rel_result:
                relationship_types[record["relation_type"]] = record["count"]

            # Total counts
            total_entities = sum(entity_types.values())
            total_relationships = sum(relationship_types.values())

            return {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "entity_types": entity_types,
                "relationship_types": relationship_types,
                "avg_relationships_per_entity": total_relationships / max(total_entities, 1),
            }

        except Exception as e:
            logger.error(f"Failed to get entity statistics: {e}")
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {},
                "avg_relationships_per_entity": 0,
            }

    async def cleanup_old_entities(self, user_id: str, session, days_threshold: int = 90):
        """Clean up old, unused entities"""
        try:
            cutoff_time = time.time() - (days_threshold * 24 * 3600)

            # Delete old entities with no relationships
            result = await session.run(
                """
                MATCH (e:Entity {user_id: $user_id})
                WHERE e.timestamp < $cutoff_time
                AND NOT (e)-[:RELATION]-()
                DELETE e
                RETURN count(e) as deleted_count
                """,
                user_id=user_id,
                cutoff_time=cutoff_time,
            )

            record = await result.single()
            deleted_count = record["deleted_count"] if record else 0

            logger.info(f"Cleaned up {deleted_count} old entities for user {user_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old entities: {e}")
            return 0


# Global graph service instance
graph_service = GraphService()
