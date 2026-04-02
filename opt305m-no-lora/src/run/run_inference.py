from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 현재 파일 기준:
# opt350m/src/run/run_inference.py
# -> parents[2] == opt350m
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_model"


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"학습된 모델 폴더가 없습니다: {MODEL_DIR}\n"
            f"먼저 train_opt350m.py를 실행해서 모델을 저장하세요."
        )

    print("학습된 토크나이저/모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    input_text = "### 질문: 우리집 강아지 이름은?\n### 답변:"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    print("추론 실행 중...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n===== 생성 결과 =====")
    print(result)


if __name__ == "__main__":
    main()