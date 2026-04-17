from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_model_lora"

def main() -> None:
    print("베이스 모델 + LoRA 어댑터 로드 중...")

    # 1. 베이스 모델을 먼저 로드
    base_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

    # 2. LoRA 어댑터를 베이스 모델 위에 얹기
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)

    # 3. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model.eval()

    input_text = ("### Question: What is my favorite sport?\n### Answer:")
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()