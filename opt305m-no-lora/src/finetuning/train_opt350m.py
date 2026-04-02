from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
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
# → parents[2] == opt350m (프로젝트 루트)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 학습된 모델 저장 위치
OUTPUT_DIR = PROJECT_ROOT / "trained_model"


# ===============================
# 🔹 학습 데이터 구성
# ===============================
def build_dataset() -> Dataset:
    data = {
        "text": [
            # [포인트] Instruction 형태로 학습
            # → 질문/답변 패턴을 모델이 학습하게 됨
            "### 질문: 내가 좋아하는 스포츠는?\n### 답변: 골프",
        ]
    }
    return Dataset.from_dict(data) # 허깅페이스 데이터셋 객체로 변환.


# ===============================
# 🔹 토큰화 함수
# ===============================
def tokenize_function(example: dict, tokenizer: AutoTokenizer) -> dict:
    return tokenizer(
        example["text"],
        padding="max_length",   # 길이를 max_length로 맞춤 (짧으면 padding)
        truncation=True,        # 길면 자름
        max_length=64,          # [포인트] 고정 길이 → 실습용
    )


# ===============================
# 🔹 메인 학습 로직
# ===============================
def main() -> None:
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # [중요]
    # OPT 모델은 기본적으로 pad_token이 없음
    # → 배치 처리 시 길이 맞추기 위해 필요
    # → eos_token을 대신 사용
    tokenizer.pad_token = tokenizer.eos_token

    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID) # Causal Language Modeling용 사전 학습 모델 로드

    # pad_token_id를 명시적으로 설정하지 않으면 에러 발생 가능
    model.config.pad_token_id = tokenizer.pad_token_id

    print("데이터셋 생성 중...")
    dataset = build_dataset() # 학습 데이터 셋

    print("토큰화 진행 중...")
    tokenized_dataset = dataset.map(
        lambda example: tokenize_function(example, tokenizer)
    )

    # [포인트]
    # Trainer는 input_ids, attention_mask만 사용
    # → text 컬럼 제거
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    # ===============================
    # 🔹 Data Collator
    # ===============================
    # 배치 단위로 데이터를 묶고 tensor로 변환
    # mlm=False → Causal LM (GPT 계열 방식)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ===============================
    # 🔹 학습 설정 (Hyperparameters)
    # ===============================
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),              # 결과 저장 경로
        per_device_train_batch_size=1,           # 한번에 하나씩 학습 (소규모 실습용)
        num_train_epochs=50,                     # 전체 데이터셋을 50번 반복 / 동일 데이터 반복 학습 (암기 유도)
        logging_steps=1,                         # 매 step 로그 출력
        save_strategy="no",                      # 체크포인트 저장 안함 (단순화)
        fp16=torch.cuda.is_available(),          # GPU 있을 경우 FP16(float16) 사용 (True시, 메모리효율이 높아짐, CPU에서는 false 유지.)
        report_to="none",                        # 외부 로깅 비활성화 (wandb 등)
    )

    # ===============================
    # 🔹 Trainer 구성
    # ===============================
    trainer = Trainer(
        model=model, # 학습할 모델
        args=training_args, # 학습 설정
        train_dataset=tokenized_dataset, # 학습 데이터셋
        processing_class=tokenizer,   # tokenizer 전달 (로그/디코딩 용)
        data_collator=data_collator, # 배치 전처리 콜레이터
    )

    print("학습 시작...")
    trainer.train()

    # ===============================
    # 🔹 모델 저장
    # ===============================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("학습된 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("학습 완료")
    print(f"저장 위치: {OUTPUT_DIR}")


# ===============================
# 🔹 실행 진입점
# ===============================
if __name__ == "__main__":
    main()