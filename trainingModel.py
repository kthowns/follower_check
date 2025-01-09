import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader

print("Data loading...")
# 1. 데이터 로드
df = pd.read_csv('IDData.csv', header=None, names=['id', 'label'])
print("Data loaded.")

print("Preparing data...")
# 2. ID 전처리
def preprocess_id(id):
    id = id.lower().strip()  # 소문자 변환 및 공백 제거
    return id

df['id'] = df['id'].apply(preprocess_id)

# 3. 레이블 인코딩 (y -> 1, n -> 0)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 4. 텍스트를 숫자 벡터로 변환 (문자 벡터화 개선)
def text_to_vector(text, max_len=30):
    # 최대 길이로 자르고, 빈 공간을 패딩
    text = text[:max_len].ljust(max_len)
    # 각 문자를 아스키 코드로 변환하고 정규화
    return [ord(c) / 255.0 for c in text]  # 정규화를 추가하여 아스키 값을 255로 나누기

# 5. 데이터셋 준비
X = df['id'].apply(text_to_vector)
y = df['label']

X = torch.tensor(X.tolist(), dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.long)

# 6. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 8. 데이터로더 준비
batch_size = 64  # 배치 크기 증가
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 9. 모델 설계 (배치 정규화 추가)
class InstagramIDModel(nn.Module):
    def __init__(self, input_size=30, hidden_size=128, output_size=2):  # hidden_size 증가
        super(InstagramIDModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 입력 레이어
        self.bn1 = nn.BatchNorm1d(hidden_size)  # 배치 정규화 추가
        self.fc2 = nn.Linear(hidden_size, output_size)  # 출력 레이어
        self.relu = nn.ReLU()  # ReLU 활성화 함수

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))  # 배치 정규화 + ReLU
        x = self.fc2(x)  # 출력층
        return x  # Softmax 제거 (CrossEntropyLoss가 Softmax를 포함)

# 10. 모델 인스턴스화
model = InstagramIDModel()

# 11. 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 낮은 학습률로 시작

print("Model training...")
# 12. 배치 단위 훈련
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # 순전파
        outputs = model(inputs)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 역전파
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("Model training done.")

print("Model Evaluating")
# 13. 모델 평가
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 14. 모델 저장 (JIT 사용)
model_scripted = torch.jit.script(model)  # 모델을 TorchScript로 변환
model_scripted.save('id_check_model_jit.pt')  # JIT 모델 저장
print("Save done.")