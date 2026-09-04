INTENT_CONFIG = {
    "disease_symptom": {
        "label": "疾病症状",
        "relation": "HAS_SYMPTOM",
        "source_type": "Disease",
        "target_type": "Symptom",
        "keywords": ("症状", "表现", "征兆", "不舒服"),
        "answer_prefix": "常见相关症状包括",
    },
    "disease_drug": {
        "label": "疾病用药",
        "relation": "HAS_DRUG",
        "source_type": "Disease",
        "target_type": "Drug",
        "keywords": ("吃什么药", "用药", "药物", "药", "治疗药"),
        "answer_prefix": "知识库收录的相关药物包括",
    },
    "disease_food": {
        "label": "饮食建议",
        "relation": "RECOMMEND_FOOD",
        "source_type": "Disease",
        "target_type": "Food",
        "keywords": ("吃什么", "饮食", "食物", "忌口", "能不能吃"),
        "answer_prefix": "知识库建议关注的饮食包括",
    },
    "disease_check": {
        "label": "检查项目",
        "relation": "NEEDS_CHECK",
        "source_type": "Disease",
        "target_type": "Check",
        "keywords": ("检查", "检验", "化验", "查什么"),
        "answer_prefix": "知识库收录的相关检查包括",
    },
    "disease_treatment": {
        "label": "治疗方式",
        "relation": "HAS_TREATMENT",
        "source_type": "Disease",
        "target_type": "Treatment",
        "keywords": ("怎么治疗", "治疗", "疗法", "处理", "改善"),
        "answer_prefix": "知识库收录的处理方式包括",
    },
    "disease_department": {
        "label": "就诊科室",
        "relation": "BELONGS_TO_DEPARTMENT",
        "source_type": "Disease",
        "target_type": "Department",
        "keywords": ("挂什么科", "哪个科", "科室", "就诊", "看什么医生"),
        "answer_prefix": "通常可优先咨询",
    },
    "drug_effect": {
        "label": "药物作用",
        "relation": "HAS_EFFECT",
        "source_type": "Drug",
        "target_type": "Effect",
        "keywords": ("作用", "功效", "有什么用", "用途"),
        "answer_prefix": "知识库收录的作用是",
    },
    "drug_usage": {
        "label": "药物用法",
        "relation": "HAS_USAGE",
        "source_type": "Drug",
        "target_type": "Usage",
        "keywords": ("怎么用", "用法", "用量", "剂量", "服用"),
        "answer_prefix": "知识库收录的用法是",
    },
}

ALLOWED_ENTITY_TYPES = {
    "Disease",
    "Symptom",
    "Drug",
    "Food",
    "Check",
    "Treatment",
    "Department",
    "Effect",
    "Usage",
}

ALLOWED_RELATIONS = {config["relation"] for config in INTENT_CONFIG.values()}

