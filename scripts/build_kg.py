import json

from app.core.config import get_settings
from app.kg.neo4j_client import Neo4jClient
from app.kg.repository import KnowledgeRepository


def main() -> None:
    settings = get_settings()
    repository = KnowledgeRepository(Neo4jClient(settings), "data/seed_knowledge.json")
    repository.initialize()
    print(json.dumps(repository.stats(), ensure_ascii=False, indent=2))
    repository.client.close()


if __name__ == "__main__":
    main()

