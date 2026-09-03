##🚀 YOLOv8 기반 Occlusion 검출 시스템 개발
#1. 📌 프로젝트 개요
🎯 목표
본 프로젝트는 딥러닝 기반 객체 탐지 모델인 YOLOv8을 활용하여 이미지 내 얼굴의 Occlusion(가림 현상)을 자동으로 검출하는 시스템을 개발하는 것을 목표로 합니다. 구체적으로는 'face', 'hat', 'sunglasses', 'accessories' 4가지 클래스를 탐지합니다. 이를 통해 얼굴 인식, 보안, 고객 행동 분석 등 다양한 응용 분야에서 Occlusion 처리에 기여할 수 있습니다.

🛠️ 사용 기술
프레임워크: Python, PyTorch, Ultralytics YOLOv8

라이브러리: Pandas, NumPy, Matplotlib, OpenCV

개발 환경: Google Colab (GPU 가속), Google Drive (데이터 및 모델 저장)

#2. 📂 데이터셋
📊 데이터 출처 및 구성
사용자 정의 데이터셋을 활용하여 Occlusion 감지용 이미지를 수집하고 어노테이션했습니다. 데이터셋은 'face', 'hat', 'sunglasses', 'accessories' 클래스에 대한 어노테이션이 포함된 이미지들로 구성되어 있습니다.

⚙️ 데이터 전처리
데이터셋은 Google Drive에 업로드 후 Colab 환경에 마운트하여 사용했습니다.

data.yaml 파일을 통해 학습, 검증, 테스트 데이터의 경로 및 클래스 정보를 YOLOv8 모델에 전달했습니다.

data.yaml에는 4개의 클래스(face, hat, sunglasses, accessories)가 정의되어 있습니다.

#3. 🧠 모델 학습 및 튜닝
🏗️ 모델 아키텍처
YOLOv8n (nano 버전) 모델을 기반으로 학습을 진행했습니다. YOLOv8은 속도와 정확도 면에서 균형 잡힌 성능을 제공하여 실시간 Occlusion 검출에 적합합니다.

🔄 학습 과정
json_to_yolo.py 스크립트를 사용하여 JSON 형식의 어노테이션 파일을 YOLO .txt 형식으로 변환했습니다.

50 에포크 (epochs=50)로 모델을 학습시켰으며, 이미지 사이즈 imgsz=640, 배치 사이즈 batch=16 등의 하이퍼파라미터를 사용했습니다.

📈 학습 결과
최종 재학습 후 모델은 다음과 같은 성능을 보였습니다:

총 mAP50-95: 0.295

총 Precision: 0.828

총 Recall: 0.363

총 mAP50: 0.543

클래스별 성능:

face: Precision: 0.895, Recall: 0.875, mAP50: 0.949, mAP50-95: 0.679

hat: Precision: 1.000, Recall: 0.000, mAP50: 0.630, mAP50-95: 0.155

sunglasses: Precision: 1.000, Recall: 0.000, mAP50: 0.110, mAP50-95: 0.058

accessories: Precision: 0.418, Recall: 0.578, mAP50: 0.484, mAP50-95: 0.290

#4. 📊 결과 분석 및 시각화
📉 학습 지표 그래프
학습 과정에서 생성된 results.png 파일을 통해 Precision, Recall, mAP, Loss 값의 변화를 시각적으로 확인했습니다. 그래프는 학습이 진행됨에 따라 지표가 안정적으로 개선되었음을 보여줍니다.

🔲 혼동 행렬 (Confusion Matrix)
confusion_matrix.png 파일을 통해 모델의 클래스별 분류 성능을 확인했습니다. 대각선 값이 높게 나타날수록 모델이 해당 클래스를 올바르게 식별하고 있음을 의미합니다.

🔍 예측 결과 예시
학습된 모델을 사용하여 새로운 이미지에 대한 Occlusion 탐지 예측을 수행했습니다. 아래 이미지는 모델이 Occlusion을 성공적으로 감지하고 바운딩 박스와 함께 클래스 및 신뢰도 점수를 표시하는 예시입니다.

#5. 💡 결론 및 향후 계획
🌱 프로젝트를 통해 배운 점
YOLOv8 모델을 활용한 객체 탐지 파이프라인 구축 및 학습 방법을 익혔습니다.

올바른 어노테이션 생성과 data.yaml 설정의 중요성을 이해했습니다.

Google Colab 및 Google Drive 연동을 통해 데이터셋 및 모델을 효율적으로 관리하는 방법을 경험했습니다.

🚀 향후 개선 방안
데이터 증강: 더 다양한 Occlusion 유형 및 환경 데이터를 추가하여 모델의 일반화 성능을 향상시킬 수 있습니다.

모델 경량화 및 최적화: 실제 생산 환경에 배포를 고려하여 모델의 크기를 줄이고 추론 속도를 최적화하는 방안을 모색할 수 있습니다.

실시간 시스템 구축: 엣지 디바이스에 모델을 배포하여 실시간 Occlusion 감지 시스템을 구현할 수 있습니다.

#💾 6. 모델 저장 및 접근
최종 학습된 모델 (best.pt)은 Google Drive의 /content/drive/MyDrive/yolo_occlusion_models_v2/yolo_occlusion_v2.pt 경로에 백업 저장되었습니다.
