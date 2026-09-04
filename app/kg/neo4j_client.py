import logging
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.driver: Any | None = None
        self.available = False

    def connect(self) -> bool:
        if not self.settings.neo4j_enabled:
            return False
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                connection_timeout=3,
            )
            self.driver.verify_connectivity()
            self.available = True
            return True
        except Exception as exc:
            logger.warning("Neo4j unavailable, using the local knowledge snapshot: %s", exc)
            self.driver = None
            self.available = False
            return False

    def ensure_schema(self, entity_types: set[str]) -> None:
        if not self.available or self.driver is None:
            return
        with self.driver.session() as session:
            for entity_type in sorted(entity_types):
                index_name = f"entity_name_{entity_type.lower()}"
                query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (node:{entity_type}) ON (node.name)"
                session.run(query).consume()

    def seed(self, entities: list[dict], relations: list[dict]) -> None:
        if not self.available or self.driver is None:
            return
        with self.driver.session() as session:
            for entity in entities:
                entity_type = entity["type"]
                query = f"""
                MERGE (node:{entity_type} {{name: $name}})
                SET node.description = $description, node.aliases = $aliases
                """
                session.run(
                    query,
                    name=entity["name"],
                    description=entity.get("description", ""),
                    aliases=entity.get("aliases", []),
                ).consume()
            for relation in relations:
                relation_type = relation["relation"]
                query = f"""
                MATCH (source {{name: $source}})
                MATCH (target {{name: $target}})
                WHERE $source_type IN labels(source) AND $target_type IN labels(target)
                MERGE (source)-[rel:{relation_type}]->(target)
                SET rel.source_type = $source_type, rel.target_type = $target_type
                """
                session.run(query, **relation).consume()

    def retrieve(self, config: dict, entity: str, limit: int = 20) -> list[dict]:
        if not self.available or self.driver is None:
            return []
        source_type = config["source_type"]
        target_type = config["target_type"]
        relation = config["relation"]
        query = f"""
        MATCH (source:{source_type} {{name: $entity}})-[rel:{relation}]->(target:{target_type})
        RETURN source.name AS entity, type(rel) AS relation, target.name AS value,
               target.description AS description
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, entity=entity, limit=limit)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("Neo4j retrieval failed: %s", exc)
            return []

    def stats(self) -> dict[str, Any] | None:
        if not self.available or self.driver is None:
            return None
        try:
            with self.driver.session() as session:
                node_count = session.run("MATCH (node) RETURN count(node) AS count").single()["count"]
                relation_count = session.run("MATCH ()-[rel]->() RETURN count(rel) AS count").single()["count"]
                node_types = session.run(
                    "MATCH (node) UNWIND labels(node) AS label RETURN label, count(*) AS count ORDER BY count DESC"
                )
                relation_types = session.run(
                    "MATCH ()-[rel]->() RETURN type(rel) AS relation, count(*) AS count ORDER BY count DESC"
                )
                return {
                    "nodes": int(node_count),
                    "relations": int(relation_count),
                    "node_types": {record["label"]: int(record["count"]) for record in node_types},
                    "relation_types": {record["relation"]: int(record["count"]) for record in relation_types},
                }
        except Exception as exc:
            logger.warning("Neo4j stats failed: %s", exc)
            return None

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None
        self.available = False

