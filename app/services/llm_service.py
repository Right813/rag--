import json
import logging
from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.kg.query_templates import INTENT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer: str
    grounded: bool
    model: str


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def status(self) -> str:
        return "configured" if self.settings.llm_configured else "grounded-fallback"

    def generate(self, question: str, intent: dict, entities: list[dict], evidence: list[dict]) -> AnswerResult:
        fallback = self._fallback_answer(question, intent, entities, evidence)
        if not evidence or not self.settings.llm_configured:
            return fallback
        try:
            answer = self._call_model(question, intent, entities, evidence)
            if self._is_grounded_answer(answer, evidence):
                return AnswerResult(answer=answer, grounded=True, model=self.settings.llm_model)
        except Exception as exc:
            logger.warning("LLM generation failed, using grounded fallback: %s", exc)
        return fallback

    def _call_model(self, question: str, intent: dict, entities: list[dict], evidence: list[dict]) -> str:
        endpoint = self.settings.llm_base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        knowledge = "\n".join(
            f"- {item['entity']} --{item['relation']}--> {item['value']}"
            for item in evidence
        )
        system_prompt = (
            "你是企业知识库问答助手。回答事实性问题时只能使用给定知识库事实，不能补充未提供的医学事实。"
            "如果知识不足必须明确说明。请用简洁中文回答，并保留关键知识值。"
        )
        payload = {
            "model": self.settings.llm_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"问题：{question}\n意图：{intent['label']}\n知识库事实：\n{knowledge}",
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return self._clean_model_content(content)

    @staticmethod
    def _clean_model_content(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
                return parsed["answer"].strip()
        except json.JSONDecodeError:
            pass
        return text

    @staticmethod
    def _is_grounded_answer(answer: str, evidence: list[dict]) -> bool:
        if not answer or len(answer) > 2000:
            return False
        return any(item["value"] in answer for item in evidence)

    def _fallback_answer(self, question: str, intent: dict, entities: list[dict], evidence: list[dict]) -> AnswerResult:
        conversational_answers = {
            "greeting": "你好！我是知库智答，可以帮你检索和解读知识库中的医疗知识与企业文档。你可以直接问我，例如“高血压有哪些常见症状？”",
            "thanks": "不客气！如果你还有疾病、药物、检查或知识库文档相关问题，随时可以继续问我。",
            "farewell": "再见！需要查询知识库时，随时回来。",
            "help": "我可以帮你查询知识库中的疾病症状、用药、检查、治疗和企业文档。你也可以上传文档后，直接询问其中的内容。",
        }
        if intent.get("name") in conversational_answers:
            return AnswerResult(
                answer=conversational_answers[intent["name"]],
                grounded=False,
                model="conversation",
            )
        if not entities:
            return AnswerResult(
                answer="我暂时没有识别出知识库中的明确实体。请试着输入疾病或药物名称，例如“高血压有哪些症状？”。",
                grounded=False,
                model="grounded-fallback",
            )
        if not evidence:
            entity_name = entities[0].get("canonical_name") or entities[0]["text"]
            return AnswerResult(
                answer=(
                    f"当前知识库暂未找到与“{entity_name}”相关的{intent['label']}信息。"
                    "你可以换个问题，或联系知识库管理员补充资料。"
                ),

                grounded=False,
                model="grounded-fallback",
            )
        values = []
        for item in evidence:
            if item["value"] not in values:
                values.append(item["value"])
        prefix = INTENT_CONFIG.get(intent["name"], {}).get("answer_prefix", "相关信息包括")
        answer = f"{prefix}：" + "、".join(values) + "。"
        if intent["name"] in {"disease_drug", "drug_effect", "drug_usage", "disease_treatment"}:
            answer += "药物和治疗方案请结合个人情况遵循医生或药品说明书。"
        answer += "以上内容来自当前知识库检索结果。"
        return AnswerResult(answer=answer, grounded=True, model="grounded-fallback")

def _generate_with_context(self, question: str, context: str, citations: list[dict]) -> AnswerResult:
    if not context.strip() or not citations:
        return AnswerResult(
            answer="当前知识库中没有检索到足够可靠的相关信息，因此无法给出确定答案。",
            grounded=False,
            model="no-answer",
        )
    fallback = _context_fallback(context)
    if not self.settings.llm_configured:
        return fallback
    try:
        answer = _call_context_model(self, question, context, citations)
        if _context_answer_is_grounded(answer, citations):
            return AnswerResult(answer=answer, grounded=True, model=self.settings.llm_model)
    except Exception as exc:
        logger.warning("Context generation failed, using grounded fallback: %s", exc)
    return fallback


def _call_context_model(self, question: str, context: str, citations: list[dict]) -> str:
    endpoint = self.settings.llm_base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    citation_text = "\n".join(
        f"[{index}] {item.get('document', '')} / 第{item.get('page')}页 / {item.get('section', '')}"
        for index, item in enumerate(citations, start=1)
    )
    system_prompt = (
        "你是企业内部文档知识库助手。只能依据给定上下文回答，不得补充上下文之外的事实。"
        "如果证据不足，请明确说明信息不足。回答简洁中文，并在相关句子后使用 [1]、[2] 标注来源。"
    )
    payload = {
        "model": self.settings.llm_model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"问题：{question}\n可用上下文：\n{context}\n来源：\n{citation_text}"},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if self.settings.llm_api_key:
        headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
    with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return self._clean_model_content(content)


def _context_answer_is_grounded(answer: str, citations: list[dict]) -> bool:
    if not answer or len(answer) > 2000:
        return False
    return any(f"[{index}]" in answer for index in range(1, len(citations) + 1))


def _context_fallback(context: str) -> AnswerResult:
    blocks = [item.strip() for item in context.split("\n\n") if item.strip()][:3]
    answer = "根据知识库检索结果：\n" + "\n\n".join(blocks)
    return AnswerResult(answer=answer[:2000], grounded=True, model="grounded-fallback")


LLMService.generate_with_context = _generate_with_context

