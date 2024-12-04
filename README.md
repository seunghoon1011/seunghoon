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
3. [코드][#코드]
4. [코드 설명](#코드 설명)
5. [라이선스](#라이선스)

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

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from matplotlib import pyplot as plt
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

# Cityscapes 데이터셋 클래스 정의
class CityscapesDataset(Dataset):
    """
    Cityscapes 데이터셋을 처리하기 위한 클래스.
    이미지 및 레이블 경로를 매칭하고, 데이터 전처리를 수행.
    """
    def __init__(self, image_dir, mask_dir, num_classes, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform
        self.image_paths, self.mask_paths = self.get_image_and_mask_paths()

    def get_image_and_mask_paths(self):
        """이미지와 마스크 경로를 매칭."""
        image_paths, mask_paths = [], []
        for city in os.listdir(self.image_dir):
            city_img_dir = os.path.join(self.image_dir, city)
            city_mask_dir = os.path.join(self.mask_dir, city)
            for file_name in os.listdir(city_img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    image_paths.append(os.path.join(city_img_dir, file_name))
                    mask_paths.append(os.path.join(city_mask_dir, file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')))
        return image_paths, mask_paths

    def preprocess_mask(self, mask):
        """레이블의 클래스 수 초과값을 제거."""
        mask[mask >= self.num_classes] = 0
        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """이미지와 레이블을 로드하고 전처리."""
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)

        mask = self.preprocess_mask(mask)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


# UNet 모델 정의
class UNet(nn.Module):
    """
    U-Net 기반 세그멘테이션 모델.
    ResNet34 백본을 사용하여 인코더-디코더 구조 구현.
    """
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.enc1 = nn.Sequential(*list(self.encoder.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x


# 모델 로드 함수
def get_model(model_type, num_classes):
    """
    모델 타입에 따라 U-Net, ResNet50, DeepLabV3+ 초기화.
    """
    if model_type == "unet":
        return UNet(num_classes)
    elif model_type == "resnet":
        model = fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        return model
    elif model_type == "deeplabv3":
        model = deeplabv3_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        return model
    else:
        raise ValueError("Invalid model type")


# 학습 루프
def train_model(model, train_loader, criterion, optimizer, num_epochs, model_name):
    """
    모델 학습 루프. 데이터셋을 반복하여 손실 최소화.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)['out'] if model_name != "unet" else model(images)
            outputs = nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")


# 시각화 함수
def visualize_predictions(model, dataloader, num_samples, model_name, device):
    """
    테스트 데이터셋에서 예측 결과를 시각화.
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    axes = axes if num_samples > 1 else [axes]

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out'] if model_name != "unet" else model(images)
            outputs = torch.argmax(outputs, dim=1).cpu().numpy()

            image = images[0].cpu().permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            image = np.clip(image, 0, 1)

            axes[i][0].imshow(image)
            axes[i][0].set_title("Input Image")
            axes[i][1].imshow(masks[0].cpu(), cmap="gray")
            axes[i][1].set_title("Ground Truth")
            axes[i][2].imshow(outputs[0], cmap="gray")
            axes[i][2].set_title(f"{model_name} Prediction")

    plt.tight_layout()
    plt.show()


# 하이퍼파라미터 및 데이터 경로 설정
image_dir = "/content/drive/MyDrive/gtFine_trainvaltest/leftImg8bit/train"
mask_dir = "/content/drive/MyDrive/gtFine_trainvaltest/gtFine/train"
num_classes = 20
batch_size = 8
learning_rate = 1e-4
num_epochs = 10

# 데이터셋 및 DataLoader 생성
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CityscapesDataset(image_dir, mask_dir, num_classes, transform=transform)
dataset_size = len(dataset)
train_dataset, test_dataset = random_split(dataset, [dataset_size // 2, dataset_size - dataset_size // 2])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# 모델 학습 및 저장
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_name in ["unet", "resnet", "deeplabv3"]:
    print(f"Training {model_name.upper()}...")
    model = get_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs, model_name)
    torch.save(model.state_dict(), f"{model_name}_model.pth")

# 모델 로드 및 결과 시각화
for model_name in ["unet", "resnet", "deeplabv3"]:
    print(f"Visualizing {model_name.upper()}...")
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(f"{model_name}_model.pth"))

    visualize_predictions(model, test_loader, num_samples=5, model_name=model_name, device=device)


# 성능 평가 실행
for model_name in ["unet", "resnet", "deeplabv3"]:
    print(f"Evaluating {model_name.upper()}...")

    # 모델 로드
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(f"{model_name}_modell.pth"))

    # 성능 평가
    metrics = evaluate_model(model, test_loader, model_name, device, num_classes)

    # 각 모델별 결과 출력
    print(f"{model_name.upper()} Metrics: {metrics}")

import pandas as pd

def evaluate_model_to_table(model, dataloader, model_name, device, num_classes):
    model.eval()  # 평가 모드로 전환
    iou_scores = []
    pixel_accuracies = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            # 모델 예측값 생성
            if model_name == "unet":
                outputs = model(images)  # UNet은 직접 출력
            else:
                outputs = model(images)['out']  # ResNet 및 DeepLab은 'out' 키의 값을 가져옴

            # 예측값을 Ground Truth 크기로 리사이즈
            outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            # 클래스 차원에서 argmax를 적용하여 클래스 예측값 생성
            outputs = torch.argmax(outputs, dim=1)

            # GPU에서 CPU로 데이터를 이동하고 numpy로 변환
            outputs = outputs.cpu().numpy()
            masks = masks.cpu().numpy()

            # 지표 계산
            for pred, gt in zip(outputs, masks):
                # IoU (Jaccard Score)
                iou = jaccard_score(gt.flatten(), pred.flatten(), average='weighted', labels=list(range(num_classes)))
                iou_scores.append(iou)

                # Pixel Accuracy
                pixel_acc = (pred == gt).sum() / gt.size
                pixel_accuracies.append(pixel_acc)

                # Precision, Recall, F1-Score
                precision = precision_score(gt.flatten(), pred.flatten(), average='weighted', zero_division=0)
                recall = recall_score(gt.flatten(), pred.flatten(), average='weighted', zero_division=0)
                f1 = f1_score(gt.flatten(), pred.flatten(), average='weighted', zero_division=0)

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

    # DataFrame 생성
    metrics_data = {
        "Metric": ["Mean IoU (mIoU)", "Pixel Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [
            np.mean(iou_scores),
            np.mean(pixel_accuracies),
            np.mean(precision_scores),
            np.mean(recall_scores),
            np.mean(f1_scores)
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df

# 성능 평가 실행 및 표 출력
for model_name in ["unet", "resnet", "deeplabv3"]:
    print(f"Evaluating {model_name.upper()}...")

    # 모델 로드
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(f"{model_name}_modell.pth"))

    # 성능 평가 및 결과 출력
    metrics_df = evaluate_model_to_table(model, test_loader, model_name, device, num_classes)

    print(f"\n{model_name.upper()} - Evaluation Metrics Table:")
    display(metrics_df)  # IPython/Jupyter 노트북 환경에서 표 시각화

------------------------------------------------------코드 부가설명---------------------------------------------------------------------
Cityscapes 데이터셋 클래스 정의
CityscapesDataset: PyTorch Dataset 클래스를 상속받아 Cityscapes 데이터를 불러오고 전처리합니다.

  __init__: 이미지와 마스크 경로를 받아 매칭.

  get_image_and_mask_paths: 데이터셋의 파일 이름 규칙을 따라 이미지와 레이블 경로를 매칭.

  preprocess_mask: 클래스 수 초과값을 제거.

  __getitem__: 이미지를 cv2로 읽어와 전처리(크기 조정 및 변환) 후 Tensor로 변환.

U-Net 모델 구현
UNet 클래스: 세그멘테이션에 사용되는 U-Net 아키텍처를 ResNet34를 백본(encoder)으로 활용하여 정의.
ResNet34의 마지막 두 계층을 제외해 enc1로 설정.
디코더(decoder)는 U-Net처럼 업샘플링하며 차원을 축소하는 구조.
최종 출력 크기를 bilinear 업샘플링으로 256x256으로 조정.
모델 초기화 함수

  get_model 함수:
다양한 모델을 초기화하는 함수. U-Net, FCN-ResNet50, DeepLabV3-ResNet50 세 가지를 지원.
모델별 최종 레이어를 세그멘테이션 클래스 수에 맞게 수정.
학습 루프
train_model 함수:
모델을 학습하는 루프 구현.
손실 함수는 CrossEntropyLoss를 사용.

학습:
모델이 이미지를 예측.
손실 계산.
역전파와 옵티마이저로 가중치 업데이트.
Epoch별 평균 손실을 출력.
시각화
visualize_predictions 함수:
학습된 모델이 테스트 데이터셋에서 예측한 결과를 시각화.
입력 이미지, Ground Truth(실제 레이블), 모델 예측을 나란히 비교.
성능 평가

평가지표:
IoU(Intersection over Union)
Pixel Accuracy
Precision, Recall, F1-Score
evaluate_model_to_table 함수:
테스트 데이터셋에 대해 위 지표들을 계산해 Pandas DataFrame 형식으로 반환.
IoU 및 정확도와 같은 지표는 sklearn 라이브러리를 사용해 계산.
데이터프레임에는 각 모델별 평균 지표가 포함
  데이터셋, 모델 학습 및 저장
데이터셋 경로와 하이퍼파라미터 설정:
Cityscapes 데이터 경로: Google Drive의 데이터를 사용.
train_loader와 test_loader 생성.

모델 학습:
세 모델(U-Net, FCN-ResNet50, DeepLabV3-ResNet50)을 각각 학습하고 가중치를 저장.
결과 시각화 및 평가
각 모델별 테스트 데이터를 사용하여 결과 시각화 및 평가.
결과 지표를 표로 정리하여 비교.
코드 실행의 흐름
데이터 로드 및 전처리: CityscapesDataset으로 데이터를 불러와 데이터로더 생성.
모델 정의 및 학습: get_model을 사용하여 모델 초기화 후 train_model로 학습.
학습된 모델 저장: 각 모델별 가중치를 .pth 파일로 저장.
시각화: visualize_predictions로 모델의 예측 결과를 확인.
성능 평가: evaluate_model_to_table로 테스트 데이터의 성능 지표 계산 후 표로 출력.

사용된 주요 라이브러리
PyTorch: 데이터 정의, 모델 구현 및 학습 루프.
Torchvision: ResNet 및 DeepLab 모델 로드, 데이터 변환.
OpenCV: 이미지 읽기 및 전처리.
NumPy: 수치 계산.
Matplotlib: 시각화.
scikit-learn: IoU, Precision, Recall, F1-Score 등 계산.

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

가장 잘 나온 모델은 DeePLabV3+ 모델이며, 아래 그림은 DeepLabV3+ 모델을 사용한 시각화 자료입니다.

![image](https://github.com/user-attachments/assets/2b505c19-596a-419c-aaeb-d1e4a5442195) > 원본 이미지
![image](https://github.com/user-attachments/assets/4fce52a3-0071-44d8-b313-6bcf70144c12) > 모델을 사용한 이미지

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


