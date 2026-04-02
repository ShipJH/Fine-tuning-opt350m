import torch

# 쿠다 제대로 설치 되었는지 확인하는 스크립트
def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("cuda version (torch):", torch.version.cuda)
        print("gpu count:", torch.cuda.device_count())
        print("gpu name:", torch.cuda.get_device_name(0))

        # 간단 연산이 GPU에서 되는지
        x = torch.randn(2, 3, device="cuda")
        y = x @ x.T
        print("gpu matmul ok:", y.shape)
    else:
        print("❌ CUDA not available. (PyTorch가 CPU 빌드이거나, 드라이버/CUDA 세팅 문제일 수 있음)")

if __name__ == "__main__":
    main()