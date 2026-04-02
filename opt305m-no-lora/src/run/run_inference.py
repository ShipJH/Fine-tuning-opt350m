from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# 🔹 프로젝트 루트 및 모델 경로
# ===============================
# 현재 파일 위치:
# opt350m/src/run/run_inference.py
# → parents[2] == opt350m
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

    # pad_token_id 다시 명시
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # ===============================
    # 🔹 질문 입력
    # ===============================
    input_text = "### 질문: 내가 좋아하는 스포츠는?\n### 답변:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("추론 실행 중...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n===== 생성 결과 =====")
    print(result)


if __name__ == "__main__":
    main()