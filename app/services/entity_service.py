from app.kg.repository import KnowledgeRepository


class EntityService:
    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    def extract(self, text: str) -> list[dict]:
        return self.repository.match_entities(text)

