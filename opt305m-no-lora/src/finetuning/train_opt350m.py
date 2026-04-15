from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import Dataset

# ===============================
# 🔹 사용할 사전 학습 모델
# ===============================
model_id = "gpt2-medium"

# ===============================
# 🔹 프로젝트 루트 / 저장 경로
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "trained_model"


def main():
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2는 pad_token이 없어서 필요

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("데이터셋 생성 중...")
    data = {
        "text": [
            f"### 질문: 내가 좋아하는 스포츠는?\n### 답변: 골프",
        ]
    }
    dataset = Dataset.from_dict(data)

    def tokenize(example):
        encoded = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        labels = encoded["input_ids"].copy()
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
        encoded["labels"] = labels
        return encoded

    print("토큰화 진행 중...")
    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        num_train_epochs=10,
        learning_rate=5e-5,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        bf16=False,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="constant",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    print("학습 시작...")
    trainer.train()

    import torch
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN 발견: {name}")
            break
    else:
        print("NaN 없음")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("학습된 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("학습 완료")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()