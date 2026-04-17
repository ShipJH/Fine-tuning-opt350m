from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# ===============================
# 🔹 사용할 사전 학습 모델
# ===============================
model_id = "gpt2-medium"

# ===============================
# 🔹 프로젝트 루트 / 저장 경로
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "trained_model_lora"

def main():
    print("시작")

    print("토크나이저 로드")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA 설정 적용
    lora_config = LoraConfig(
        r=16,  # 🔹 LoRA의 "랭크(rank)" 값입니다.
              # 학습할 파라미터 수를 줄이는 정도를 설정합니다.
              # r이 작을수록 계산이 가벼워지지만, 너무 작으면 성능이 떨어질 수 있습니다.

        lora_alpha=16,  # 🔹 LoRA가 학습한 정보를 얼마나 강하게 모델에 반영할지 정하는 값입니다.
                        # 일종의 "확대 비율"처럼 작용하며, 일반적으로 r과 함께 조정합니다.

        lora_dropout=0.05,  # 🔹 학습 중 일부 정보를 무작위로 버려 과적합을 막는 기술입니다.
                            # 0.05는 5% 확률로 드롭아웃이 일어나도록 설정한 것입니다.

        bias="none",  # 🔹 기존 모델의 편향(bias) 파라미터는 건드리지 않겠다는 뜻입니다.
                      # 즉, 오직 LoRA 레이어만 학습합니다.

        task_type="CAUSAL_LM"  # 🔹 이 설정이 적용될 작업의 유형입니다.
                               # "CAUSAL_LM"은 일반적인 언어 생성 모델(예: GPT)에서 사용됩니다.
    )

    model = get_peft_model(base_model, lora_config)


    print("데이터셋 생성 중...")
    data = {
        "text": [
            f"### 질문: 내가 좋아하는 스포츠는?\n### 답변: 골프",
        ]
    }
    dataset = Dataset.from_dict(data)

    # 토큰화
    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)

    print("토큰화 진행 중...")
    tokenized_dataset = dataset.map(tokenize)

    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 학습 하이퍼파라미터
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=150,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        report_to="none"
    )

    # Trainer 구성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("학습 시작...")
    trainer.train()

if __name__ == "__main__":
    main()