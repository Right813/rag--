from app.core.config import Settings
from app.db.redis import Cache
from app.kg.neo4j_client import Neo4jClient
from app.kg.repository import KnowledgeRepository
from app.services.entity_service import EntityService
from app.services.intent_service import IntentService


def test_entity_matcher_prefers_canonical_longest_term():
    settings = Settings(neo4j_enabled=False, redis_enabled=False)
    repository = KnowledgeRepository(Neo4jClient(settings), "data/seed_knowledge.json")
    repository.initialize()

    entities = EntityService(repository).extract("我想了解高血压病的症状")

    assert len(entities) == 1
    assert entities[0]["canonical_name"] == "高血压"
    assert entities[0]["score"] < 1


def test_intent_router_prioritizes_drug_usage():
    intent = IntentService().classify(
        "氨氯地平怎么用？",
        [{"text": "氨氯地平", "type": "Drug", "canonical_name": "氨氯地平"}],
    )

    assert intent["name"] == "drug_usage"


def test_memory_cache_round_trip():
    settings = Settings(redis_enabled=False)
    cache = Cache(settings)
    cache.initialize()
    cache.set_json("test-key", {"answer": "ok"})

    assert cache.get_json("test-key") == {"answer": "ok"}

