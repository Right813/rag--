from app.kg.query_templates import INTENT_CONFIG


class IntentService:
    def classify(self, text: str, entities: list[dict]) -> dict:
        normalized = text.casefold()
        entity_type = entities[0]["type"] if entities else None
        if entity_type == "Drug":
            if self._contains_any(normalized, INTENT_CONFIG["drug_usage"]["keywords"]):
                return self._result("drug_usage", 0.98)
            if self._contains_any(normalized, INTENT_CONFIG["drug_effect"]["keywords"]):
                return self._result("drug_effect", 0.98)
        if entity_type == "Disease":
            ordered_intents = (
                "disease_department",
                "disease_check",
                "disease_drug",
                "disease_food",
                "disease_symptom",
                "disease_treatment",
            )
            for intent_name in ordered_intents:
                if self._contains_any(normalized, INTENT_CONFIG[intent_name]["keywords"]):
                    return self._result(intent_name, 0.97)
        return self._result("general", 0.65)

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.casefold() in text for keyword in keywords)

    @staticmethod
    def _result(name: str, confidence: float) -> dict:
        if name == "general":
            return {"name": name, "label": "知识库问答", "confidence": confidence}
        return {"name": name, "label": INTENT_CONFIG[name]["label"], "confidence": confidence}

