# DACON Physical Stability Classification

DACON 이미지 기반 물체 안정성 분류 실험 저장소입니다. top/front 이미지를 함께 사용해 `stable`/`unstable` 확률을 예측하는 PyTorch 모델을 실험합니다.

## Repository Contents

- `main.ipynb`: 이미지 데이터 로딩, augmentation, two-view 모델 학습, 제출 파일 생성
- `use_videodata.ipynb`: 비디오/프레임 데이터를 활용한 추가 실험
- `requirements.txt`: 노트북 실행에 필요한 주요 Python 패키지

## Data

이 저장소는 대용량 데이터, 제출 파일, 체크포인트를 git에 포함하지 않습니다.

DACON에서 받은 데이터를 `daicon/physical/` 아래에 배치하세요.

예상 구조:

```text
daicon/
  physical/
    train.csv
    dev.csv
    sample_submission.csv
    train/
    dev/
    test/
```

학습된 모델 파일(`*.pt`, `*.pth`)과 생성된 제출 CSV는 필요 시 로컬에만 보관합니다.

## Workflow

1. 의존성 설치

```bash
pip install -r requirements.txt
```

2. 학습 노트북 실행

```bash
jupyter lab main.ipynb
```

3. 제출 파일 생성

노트북 마지막 inference 셀을 실행하면 로컬에 제출 CSV가 생성됩니다.

## Notes

- 노트북은 CUDA 환경을 기준으로 작성되어 있습니다.
- augmentation에는 Albumentations를 사용합니다.
- 현재 로컬 파일명에는 `submisson.csv`처럼 오타가 있는 제출 파일이 포함되어 있지만, git에는 제외됩니다.
