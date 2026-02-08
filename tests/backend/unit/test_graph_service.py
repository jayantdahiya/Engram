"""Unit tests for GraphService (Neo4j operations)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.graph_service import GraphService


class TestGraphService:
    """Test cases for GraphService."""

    @pytest.fixture
    def graph_service(self):
        """Create graph service instance with mocked driver."""
        with patch("services.graph_service.settings") as mock_settings:
            mock_settings.neo4j_uri = "bolt://localhost:7687"
            mock_settings.neo4j_user = "neo4j"
            mock_settings.neo4j_password = "password"
            service = GraphService()
            service.driver = AsyncMock()
            return service

    @pytest.fixture
    def mock_neo4j_session(self):
        """Create mock Neo4j session."""
        session = AsyncMock()
        session.run = AsyncMock()
        return session

    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            {
                "id": "john-doe",
                "name": "John",
                "type": "PERSON",
                "attributes": {"age": 30},
            },
            {
                "id": "acme-corp",
                "name": "Acme Corp",
                "type": "ORGANIZATION",
                "attributes": {"industry": "tech"},
            },
        ]

    @pytest.fixture
    def sample_relationships(self):
        """Sample relationships for testing."""
        return [
            {
                "source": "john-doe",
                "target": "acme-corp",
                "relation": "WORKS_AT",
                "attributes": {},
            },
        ]

    @pytest.mark.asyncio
    async def test_initialize(self, graph_service):
        """Test Neo4j driver initialization."""
        with patch("services.graph_service.AsyncGraphDatabase") as mock_db:
            mock_db.driver.return_value = AsyncMock()

            await graph_service.initialize()

            mock_db.driver.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, graph_service):
        """Test Neo4j driver close."""
        graph_service.driver = AsyncMock()

        await graph_service.close()

        graph_service.driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_graph_memory(
        self, graph_service, mock_neo4j_session, sample_entities, sample_relationships
    ):
        """Test storing entities and relationships."""
        mock_neo4j_session.run.return_value = MagicMock()

        await graph_service.store_graph_memory(
            entities=sample_entities,
            relationships=sample_relationships,
            user_id="test-user",
            session=mock_neo4j_session,
        )

        # Should have called run for entities and relationships
        assert mock_neo4j_session.run.call_count >= 1

    @pytest.mark.asyncio
    async def test_store_graph_memory_empty(self, graph_service, mock_neo4j_session):
        """Test storing with empty entities/relationships."""
        await graph_service.store_graph_memory(
            entities=[],
            relationships=[],
            user_id="test-user",
            session=mock_neo4j_session,
        )

        # Should complete without error

    @pytest.mark.asyncio
    async def test_get_entity_relationships(self, graph_service, mock_neo4j_session):
        """Test retrieving entity relationships."""
        # Create async iterator for result
        mock_records = [
            {
                "path": MagicMock(
                    nodes=[
                        {"id": "John", "name": "John", "type": "PERSON"},
                        {"id": "Acme", "name": "Acme", "type": "ORG"},
                    ],
                    relationships=[
                        MagicMock(type="WORKS_AT", get=lambda k, d=None: 1.0)
                    ],
                ),
                "depth": 1,
            }
        ]

        async def async_gen():
            for record in mock_records:
                yield record

        mock_result = MagicMock()
        mock_result.__aiter__.side_effect = async_gen
        mock_neo4j_session.run.return_value = mock_result

        result = await graph_service.get_entity_relationships(
            entity_id="John",
            user_id="test-user",
            session=mock_neo4j_session,
            max_depth=2,
        )

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_related_entities(self, graph_service, mock_neo4j_session):
        """Test finding related entities from query."""
        mock_records = [
            {
                "id": "Acme",
                "name": "Acme Corp",
                "type": "ORGANIZATION",
                "relationship_types": ["WORKS_AT"],
                "confidences": [1.0],
            }
        ]

        async def async_gen():
            for record in mock_records:
                yield record

        mock_result = MagicMock()
        mock_result.__aiter__.side_effect = async_gen
        mock_neo4j_session.run.return_value = mock_result

        result = await graph_service.find_related_entities(
            query_entities=["John"],
            user_id="test-user",
            session=mock_neo4j_session,
        )

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_related_entities_with_relationship_filter(
        self, graph_service, mock_neo4j_session
    ):
        """Test finding related entities with relationship type filter."""

        async def async_gen():
            if False:
                yield  # Empty generator

        mock_result = MagicMock()
        mock_result.__aiter__.side_effect = async_gen
        mock_neo4j_session.run.return_value = mock_result

        result = await graph_service.find_related_entities(
            query_entities=["John"],
            user_id="test-user",
            session=mock_neo4j_session,
            relationship_types=["WORKS_AT"],
        )

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_user_entity_graph(self, graph_service, mock_neo4j_session):
        """Test retrieving complete user entity graph."""
        # Setup entities result
        mock_entities = [
            {"id": "John", "name": "John", "type": "PERSON", "timestamp": 123}
        ]

        async def async_gen_entities():
            for r in mock_entities:
                yield r

        mock_entities_result = MagicMock()
        mock_entities_result.__aiter__.side_effect = async_gen_entities

        # Setup relationships result
        mock_rels = [
            {
                "source": "John",
                "target": "Acme",
                "relation": "WORKS_AT",
                "confidence": 1.0,
            }
        ]

        async def async_gen_rels():
            for r in mock_rels:
                yield r

        mock_rels_result = MagicMock()
        mock_rels_result.__aiter__.side_effect = async_gen_rels

        # Configure session.run to return different results
        mock_neo4j_session.run.side_effect = [mock_entities_result, mock_rels_result]

        result = await graph_service.get_user_entity_graph(
            user_id="test-user",
            session=mock_neo4j_session,
            limit=100,
        )

        assert isinstance(result, dict)
        assert len(result["entities"]) == 1
        assert len(result["relationships"]) == 1

    @pytest.mark.asyncio
    async def test_search_entities_by_name(self, graph_service, mock_neo4j_session):
        """Test searching entities by name pattern."""
        mock_records = [
            {"id": "John", "name": "John Doe", "type": "PERSON", "confidence": 1.0}
        ]

        async def async_gen():
            for r in mock_records:
                yield r

        mock_result = MagicMock()
        mock_result.__aiter__.side_effect = async_gen
        mock_neo4j_session.run.return_value = mock_result

        result = await graph_service.search_entities_by_name(
            name_pattern="John",
            user_id="test-user",
            session=mock_neo4j_session,
            limit=20,
        )

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_entity_statistics(self, graph_service, mock_neo4j_session):
        """Test getting entity statistics."""
        # Entities types
        mock_types = [{"type": "PERSON", "count": 5}, {"type": "ORG", "count": 5}]

        async def async_gen_types():
            for r in mock_types:
                yield r

        mock_types_result = MagicMock()
        mock_types_result.__aiter__.side_effect = async_gen_types

        # Rel types
        mock_rels = [{"relation_type": "WORKS_AT", "count": 10}]

        async def async_gen_rels():
            for r in mock_rels:
                yield r

        mock_rels_result = MagicMock()
        mock_rels_result.__aiter__.side_effect = async_gen_rels

        mock_neo4j_session.run.side_effect = [mock_types_result, mock_rels_result]

        result = await graph_service.get_entity_statistics(
            user_id="test-user",
            session=mock_neo4j_session,
        )

        assert isinstance(result, dict)
        assert result["total_entities"] == 10
        assert result["total_relationships"] == 10

    @pytest.mark.asyncio
    async def test_cleanup_old_entities(self, graph_service, mock_neo4j_session):
        """Test cleaning up old entities."""
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"deleted_count": 3})
        mock_neo4j_session.run.return_value = mock_result

        await graph_service.cleanup_old_entities(
            user_id="test-user",
            session=mock_neo4j_session,
            days_threshold=90,
        )

        mock_neo4j_session.run.assert_called()

    @pytest.mark.asyncio
    async def test_store_graph_memory_with_attributes(
        self, graph_service, mock_neo4j_session
    ):
        """Test storing entities with complex attributes."""
        entities = [
            {
                "name": "Project Alpha",
                "type": "PROJECT",
                "id": "project-alpha",  # ADDED ID
                "attributes": {
                    "status": "active",
                    "budget": 100000,
                    "tags": ["important", "urgent"],
                },
            }
        ]

        mock_neo4j_session.run.return_value = MagicMock()

        await graph_service.store_graph_memory(
            entities=entities,
            relationships=[],
            user_id="test-user",
            session=mock_neo4j_session,
        )

        assert mock_neo4j_session.run.called
