import re

from app.kg.query_templates import INTENT_CONFIG


CONVERSATIONAL_INTENTS = {"greeting", "thanks", "farewell", "help"}


class IntentService:
    def classify(self, text: str, entities: list[dict]) -> dict:
        normalized = text.casefold()
        if not entities:
            conversational_intent = self._classify_conversation(normalized)
            if conversational_intent:
                return self._result(conversational_intent, 0.99)
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
    def _classify_conversation(text: str) -> str | None:
        compact = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
        if compact in {
            "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey",
            "早上好", "上午好", "中午好", "下午好", "晚上好",
        }:
            return "greeting"
        if compact in {"谢谢", "感谢", "多谢", "谢了", "辛苦了", "thankyou", "thanks"}:
            return "thanks"
        if compact in {"再见", "拜拜", "拜拜了", "晚安", "goodbye", "bye"}:
            return "farewell"
        if compact in {
            "帮助", "帮帮我", "你能做什么", "你会做什么", "有什么功能",
            "怎么使用", "如何使用", "怎么用", "使用说明",
        }:
            return "help"
        return None

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.casefold() in text for keyword in keywords)

    @staticmethod
    def _result(name: str, confidence: float) -> dict:
        if name == "general":
            return {"name": name, "label": "知识库问答", "confidence": confidence}
        conversational_labels = {
            "greeting": "问候交流",
            "thanks": "感谢反馈",
            "farewell": "结束交流",
            "help": "使用帮助",
        }
        if name in conversational_labels:
            return {"name": name, "label": conversational_labels[name], "confidence": confidence}
        return {"name": name, "label": INTENT_CONFIG[name]["label"], "confidence": confidence}
