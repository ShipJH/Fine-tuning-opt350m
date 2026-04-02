from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

# ===============================
# 🔹 사용할 사전 학습 모델
# ===============================
model_id = "facebook/opt-350m"

# ===============================
# 🔹 프로젝트 루트 / 저장 경로
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "trained_model"

# ===============================
# 🔹 사전 학습된 모델 및 토크나이저 불러오기
# ===============================
def main():
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    tokenizer.pad_token = tokenizer.eos_token

    # OPT 모델은 'pad_token'이 기본적으로 정의되어 있지 않음
    # 그래서 eos_token을 pad_token으로 사용
    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # ===============================
    # 🔹 학습할 데이터 구성
    # ===============================
    print("데이터셋 생성 중...")
    data = {
        "text": [
            "### 질문: 내가 좋아하는 스포츠는?\n### 답변: 골프",
        ]
    }
    dataset = Dataset.from_dict(data)

    # ===============================
    # 🔹 토큰화 함수 정의 및 적용
    # ===============================
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=64
        )

    print("토큰화 진행 중...")
    # [로컬 호환성 수정 1]
    # 현재 transformers/data collator 환경에서는 원본 text 컬럼이 남아 있으면
    # 배치 생성 시 문자열을 tensor로 바꾸려다 에러가 날 수 있음
    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

    # ===============================
    # 🔹 데이터 콜레이터 설정
    # ===============================
    # 모델에 배치로 넣기 전에 텐서 형태로 묶어주는 역할
    # mlm=False → Causal LM 방식
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ===============================
    # 🔹 학습 하이퍼파라미터 설정
    # ===============================
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),  # 학습 결과 저장 폴더
        per_device_train_batch_size=1,  # [포인트] 한 번에 하나씩 학습 → 소규모 실습용 설정
        num_train_epochs=10,  # 전체 데이터셋을 50번 반복 학습
        logging_steps=1,  # 매 스텝마다 로그 출력
        save_strategy="no",  # 학습 중 체크포인트 저장하지 않음
        report_to="none",  # 로그 저장 위치 (None으로 설정 시 WandB 등 외부로 전송 안 함)
    )

    # ===============================
    # 🔹 Trainer 객체 구성 및 학습 시작
    # ===============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,   # [로컬 호환성 수정 2]
        data_collator=data_collator,
    )

    print("학습 시작...")
    trainer.train()

    # ===============================
    # 🔹 학습된 모델 저장
    # ===============================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("학습된 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("학습 완료")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()