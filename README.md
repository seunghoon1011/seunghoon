# seunghoon

## Data Preparation
Place the dataset in the following structure:
If you do not have the dataset, download it from the [Cityscapes website](https://www.cityscapes-dataset.com/) and extract it into the `data/` folder.
(https://www.cityscapes-dataset.com/leftImg8bit_trainvaltest.zip)
(https://www.cityscapes-dataset.com/gtFine_trainvaltest.zip)


# Cityscapes 세그멘테이션 프로젝트

이 프로젝트는 Cityscapes 데이터셋을 사용하여 **U-Net**, **FCN-ResNet50**, **DeepLabV3-ResNet50** 모델로 세그멘테이션을 구현합니다.

---

## 목차

1. [개요](#개요)
2. [데이터셋](#데이터셋)
3. [모델](#모델)
4. [사용 방법](#사용-방법)
5. [결과](#결과)
6. [평가](#평가)
7. [라이선스](#라이선스)

---

## 개요

이 프로젝트의 목표는 Cityscapes 데이터셋을 활용하여 도시 환경의 장면 이해를 위한 세그멘테이션을 수행하는 것입니다. 이 레포지토리는 다음과 같은 기능을 제공합니다:

- ResNet34를 백본으로 한 U-Net의 커스텀 구현
- torchvision의 사전 학습된 FCN-ResNet50 및 DeepLabV3-ResNet50 모델 사용
- IoU, 픽셀 정확도, Precision, Recall, F1-Score 등의 평가 지표
- 모델 예측 결과와 실제 레이블을 비교하여 시각화

---

## 데이터셋

이 프로젝트는 고품질의 이미지와 어노테이션이 포함된 [Cityscapes 데이터셋](https://www.cityscapes-dataset.com/)을 사용합니다.

1. **데이터셋 다운로드**
   - 이미지 데이터: [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
   - 어노테이션 데이터: [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
2. **폴더 구조**
   다운로드한 데이터를 `data/` 폴더에 아래와 같이 배치합니다:

data/ ├── leftImg8bit/ │ ├── train/ │ ├── val/ │ └── test/ ├── gtFine/ │ ├── train/ │ ├── val/ │ └── test/
yaml

---

## 모델

이 프로젝트에서는 세 가지 모델을 구현합니다:
1. **U-Net (ResNet34 백본)**
- 맞춤형 디코더 구조로 업샘플링 수행
- 학습 파라미터 수를 최적화하여 성능 개선
- ResNet34를 백본으로 사용하여 커스텀 구현되었습니다.
2. **FCN-ResNet50**
- torchvision의 사전 학습된 모델 사용
- 세그멘테이션 클래스 수에 맞게 수정
- torchvision의 사전 학습된 모델을 활용하였습니다.
3. **DeepLabV3-ResNet50**
- torchvision의 사전 학습된 모델 사용
- 고급 디코더로 세밀한 세그멘테이션 수행
- torchvision의 사전 학습된 모델을 활용하였습니다.
---


## 데이터셋 라이선스

이 프로젝트는 [Cityscapes Dataset](https://www.cityscapes-dataset.com/)을 사용합니다. 해당 데이터셋은 Cityscapes Dataset License의 적용을 받으며, 데이터 사용 시 다음 사항을 준수해야 합니다:

1. 데이터는 연구 목적으로만 사용 가능합니다.
2. 상업적 사용은 금지됩니다.
3. 데이터 재배포는 허용되지 않습니다.

자세한 내용은 [Cityscapes License](https://www.cityscapes-dataset.com/license/)를 참고하세요.

Cityscapes 데이터셋은 Cityscapes Consortium의 소유입니다. 해당 데이터셋은 다음의 라이선스를 따릅니다:
[Cityscapes Dataset License](https://www.cityscapes-dataset.com/license/)
- 연구 및 비상업적 목적으로만 사용 가능합니다.
- 데이터 재배포 및 상업적 사용은 금지됩니다.
- 데이터셋을 사용하는 경우 적절한 인용이 필요합니다:
  M. Cordts, et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR 2016.

프로젝트에서 Cityscapes 데이터셋을 사용하는 경우, 위 약관을 준수해야 합니다.

## 데이터셋 인용

@inproceedings{Cordts2016Cityscapes,
  title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
  author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}


