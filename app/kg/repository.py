import json
import logging
import re
from pathlib import Path
from typing import Any

from app.kg.neo4j_client import Neo4jClient
from app.kg.query_templates import ALLOWED_ENTITY_TYPES, ALLOWED_RELATIONS, INTENT_CONFIG

logger = logging.getLogger(__name__)
_SAFE_IDENTIFIER = re.compile(r"^[A-Z][A-Za-z0-9_]*$")


class KnowledgeRepository:
    def __init__(
        self,
        client: Neo4jClient,
        seed_path: str | Path,
        storage_path: str | Path | None = None,
    ) -> None:
        self.client = client
        self.seed_path = Path(seed_path)
        self.storage_path = Path(storage_path) if storage_path else None
        self.entities: list[dict] = []
        self.relations: list[dict] = []
        self._entity_index: dict[str, dict] = {}
        self.initialized = False

    def initialize(self) -> None:
        source_path = self.storage_path if self.storage_path and self.storage_path.exists() else self.seed_path
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        self.entities = payload.get("entities", [])
        self.relations = payload.get("relations", [])
        self._rebuild_index()
        self.client.connect()
        self.client.ensure_schema({entity["type"] for entity in self.entities})
        self.client.seed(self.entities, self.relations)
        self.initialized = True

    def _rebuild_index(self) -> None:
        self._entity_index = {}
        for entity in self.entities:
            canonical = self._normalize(entity["name"])
            self._entity_index[canonical] = entity
            for alias in entity.get("aliases", []):
                self._entity_index[self._normalize(alias)] = entity

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[\s\-_]", "", value).casefold()

    def match_entities(self, text: str) -> list[dict]:
        candidates: list[dict] = []
        seen: set[str] = set()
        searchable: list[tuple[str, dict, float]] = []
        for entity in self.entities:
            searchable.append((entity["name"], entity, 0.99))
            searchable.extend((alias, entity, 0.93) for alias in entity.get("aliases", []))
        searchable.sort(key=lambda item: len(item[0]), reverse=True)
        lower_text = text.casefold()
        occupied: list[tuple[int, int]] = []
        for term, entity, score in searchable:
            start = lower_text.find(term.casefold())
            if start < 0 or entity["name"] in seen:
                continue
            end = start + len(term)
            if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied):
                continue
            seen.add(entity["name"])
            occupied.append((start, end))
            candidates.append(
                {
                    "text": text[start:end],
                    "type": entity["type"],
                    "start": start,
                    "end": end,
                    "score": score,
                    "canonical_name": entity["name"],
                }
            )
        return sorted(candidates, key=lambda item: item["start"])

    def retrieve(self, entity: str, intent_name: str, limit: int = 20) -> list[dict]:
        config = INTENT_CONFIG.get(intent_name)
        if config is None:
            return []
        results = self.client.retrieve(config, entity, limit)
        if results:
            return [self._format_result(result) for result in results]
        entity_record = self._entity_index.get(self._normalize(entity))
        if entity_record is None:
            return []
        matches = [
            relation
            for relation in self.relations
            if relation["source"] == entity_record["name"]
            and relation["relation"] == config["relation"]
            and relation["target_type"] == config["target_type"]
        ][:limit]
        target_by_name = {item["name"]: item for item in self.entities}
        return [
            {
                "entity": relation["source"],
                "relation": relation["relation"],
                "value": relation["target"],
                "source": "medical_kg",
                "description": target_by_name.get(relation["target"], {}).get("description", ""),
            }
            for relation in matches
        ]

    @staticmethod
    def _format_result(result: dict) -> dict:
        return {
            "entity": result.get("entity", ""),
            "relation": result.get("relation", ""),
            "value": result.get("value", ""),
            "source": "medical_kg",
            "description": result.get("description") or "",
        }

    def merge(self, entities: list[dict], relations: list[dict]) -> dict[str, int]:
        known_entities = {entity["name"]: entity for entity in self.entities}
        for entity in entities:
            if entity["type"] not in ALLOWED_ENTITY_TYPES or not _SAFE_IDENTIFIER.match(entity["type"]):
                raise ValueError(f"Unsupported entity type: {entity['type']}")
            known_entities[entity["name"]] = entity
        known_names = set(known_entities)
        for relation in relations:
            if relation["relation"] not in ALLOWED_RELATIONS or not _SAFE_IDENTIFIER.match(relation["relation"]):
                raise ValueError(f"Unsupported relation: {relation['relation']}")
            if relation["source"] not in known_names or relation["target"] not in known_names:
                raise ValueError("Relation endpoints must exist in the entity set")
        self.entities = list(known_entities.values())
        existing = {(item["source"], item["relation"], item["target"]) for item in self.relations}
        for relation in relations:
            relation_key = (relation["source"], relation["relation"], relation["target"])
            if relation_key not in existing:
                self.relations.append(relation)
                existing.add(relation_key)
        self._rebuild_index()
        self.client.ensure_schema({entity["type"] for entity in self.entities})
        self.client.seed(entities, relations)
        self._persist()
        return {"nodes": len(self.entities), "relations": len(self.relations)}

    def _persist(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.storage_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps({"entities": self.entities, "relations": self.relations}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.storage_path)

    def stats(self) -> dict[str, Any]:
        graph_stats = self.client.stats()
        if graph_stats is not None:
            return graph_stats
        node_types: dict[str, int] = {}
        relation_types: dict[str, int] = {}
        for entity in self.entities:
            node_types[entity["type"]] = node_types.get(entity["type"], 0) + 1
        for relation in self.relations:
            relation_types[relation["relation"]] = relation_types.get(relation["relation"], 0) + 1
        return {
            "nodes": len(self.entities),
            "relations": len(self.relations),
            "node_types": node_types,
            "relation_types": relation_types,
        }
