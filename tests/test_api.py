def test_health_reports_ready_components(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["components"]["api"] == "ok"
    assert payload["components"]["neo4j"] == "memory-fallback"


def test_chat_returns_grounded_evidence_and_persists_history(client):
    response = client.post(
        "/api/v1/chat",
        json={"query": "高血压有哪些常见症状？", "session_id": "api-test-session"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["intent"]["name"] == "disease_symptom"
    assert "头晕" in payload["answer"]
    assert payload["evidence"][0]["source"] == "medical_kg"

    history = client.get("/api/v1/history?session_id=api-test-session")
    assert history.status_code == 200
    assert [item["role"] for item in history.json()["messages"]] == ["user", "assistant"]


def test_repeat_chat_uses_cache(client):
    first = client.post("/api/v1/chat", json={"query": "高血压需要做什么检查？"})
    second = client.post("/api/v1/chat", json={"query": "高血压需要做什么检查？"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["evidence"][0]["value"] == "血压监测"


def test_unknown_question_is_not_invented(client):
    response = client.post("/api/v1/chat", json={"query": "火星基地的维修流程是什么？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is False
    assert payload["evidence"] == []
    assert "暂时没有识别出" in payload["answer"]


def test_greeting_uses_conversational_response(client):
    response = client.post("/api/v1/chat", json={"query": "你好"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"]["name"] == "greeting"
    assert payload["grounded"] is False
    assert payload["no_answer"] is False
    assert payload["evidence"] == []
    assert payload["answer"].startswith("你好！")


def test_import_validates_and_updates_local_knowledge(client):
    response = client.post(
        "/api/v1/knowledge/import",
        json={
            "entities": [
                {"name": "偏头痛", "type": "Disease", "aliases": [], "description": "头痛类疾病。"},
                {"name": "休息", "type": "Treatment", "aliases": [], "description": "适当休息。"},
            ],
            "relations": [
                {
                    "source": "偏头痛",
                    "source_type": "Disease",
                    "relation": "HAS_TREATMENT",
                    "target": "休息",
                    "target_type": "Treatment",
                }
            ],
        },
    )

    assert response.status_code == 200
    answer = client.post("/api/v1/chat", json={"query": "偏头痛怎么治疗？"}).json()
    assert answer["grounded"] is True
    assert "休息" in answer["answer"]


def test_csv_relation_import_and_cache_invalidation(client):
    first = client.post("/api/v1/chat", json={"query": "高血压有哪些症状？"})
    assert first.status_code == 200

    csv_content = "source,source_type,relation,target,target_type\n高血压,Disease,HAS_SYMPTOM,头晕,Symptom\n"
    imported = client.post(
        "/api/v1/knowledge/import-file",
        files={"file": ("relations.csv", csv_content.encode("utf-8"), "text/csv")},
    )

    assert imported.status_code == 200
    second = client.post("/api/v1/chat", json={"query": "高血压有哪些症状？"})
    assert second.status_code == 200
    assert second.json()["cached"] is False
