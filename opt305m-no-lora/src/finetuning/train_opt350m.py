from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ===============================
# 🔹 사용할 사전 학습 모델
# ===============================
MODEL_ID = "facebook/opt-350m"

# ===============================
# 🔹 프로젝트 루트 및 저장 경로 설정
# ===============================
# 현재 파일 위치:
# opt350m/src/finetuning/train_opt350m.py
# → parents[2] == opt350m
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "trained_model"


# ===============================
# 🔹 학습할 데이터 구성
# ===============================
def build_dataset() -> Dataset:
    data = {
        "text": [
            "### 질문: 내가 좋아하는 스포츠는?\n### 답변: 골프",
        ]
    }
    return Dataset.from_dict(data)


# ===============================
# 🔹 토큰화 함수 정의
# ===============================
def tokenize(example: dict, tokenizer: AutoTokenizer) -> dict:
    # [포인트]
    # max_length=64로 고정
    # → 긴 문장은 자르고, 짧은 문장은 padding
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
    )


# ===============================
# 🔹 메인 학습 로직
# ===============================
def main() -> None:
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # OPT 모델은 기본적으로 pad_token이 없음
    # 그래서 eos_token을 pad_token으로 사용
    tokenizer.pad_token = tokenizer.eos_token

    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    # pad_token_id를 모델 설정에도 반영
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("데이터셋 생성 중...")
    dataset = build_dataset()

    print("토큰화 진행 중...")
    tokenized_dataset = dataset.map(
        lambda example: tokenize(example, tokenizer),
        remove_columns=["text"],
    )

    # ===============================
    # 🔹 데이터 콜레이터 설정
    # ===============================
    # mlm=False → Causal LM 방식
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ===============================
    # 🔹 학습 하이퍼파라미터 설정
    # ===============================
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        num_train_epochs=50,
        logging_steps=1,
        save_strategy="no",
        fp16=False,                          # 로컬에서는 일단 끔
        bf16=False,                          # 로컬에서는 일단 끔
        report_to="none",
        remove_unused_columns=False,         # 원본 스타일 유지
    )

    # ===============================
    # 🔹 Trainer 객체 구성
    # ===============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("학습 시작...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("학습된 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("학습 완료")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()