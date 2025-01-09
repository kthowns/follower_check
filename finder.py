import torch
import torch.nn.functional as F
import re

# 텍스트를 벡터로 변환하는 함수
def text_to_vector(text, max_len=30):
    # 최대 길이로 자르고, 빈 공간을 패딩
    text = text[:max_len].ljust(max_len)
    # 각 문자를 아스키 코드로 변환하고 정규화
    return [ord(c) / 255.0 for c in text]  # 정규화를 추가하여 아스키 값을 255로 나누기

def is_valid_instagram_id(id):
    # 길이가 2에서 30자 사이인지 확인
    if not (2 <= len(id) <= 30):
        return False

    # ID가 허용된 문자만 포함하는지 확인 (소문자, 숫자, 밑줄, 점)
    if not re.match("^[a-z0-9_\\.]+$", id):
        return False

    # 점이 처음이나 끝에 오지 않으며 연속해서 두 개 이상 올 수 없음
    if id.startswith('.') or id.endswith('.') or '..' in id:
        return False

    return True

def extractIdList(lines):
    resultId = []
    for line in lines:
        line = line.strip()  # 공백 제거
        input_vector = text_to_vector(line)
        input_tensor = torch.tensor([input_vector], dtype=torch.float32)  # 2D 텐서로 변환

        # 모델에 입력하여 예측
        with torch.no_grad():
            output = loaded_model(input_tensor)

        # 소프트맥스 함수로 확률로 변환
        probabilities = F.softmax(output, dim=1)

        # 예측된 클래스와 확률
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probability = probabilities[0][predicted_class].item()
        if predicted_probability < 0.89 and is_valid_instagram_id(line):
            resultId.append(line)
    return resultId

# JIT 모델 로드
loaded_model = torch.jit.load('model.pt')
loaded_model.eval()  # 평가 모드로 전환

print("팔로워 목록 파일 이름 입력 ex) follower.txt : ", end='')
followerFile = input()
with open(followerFile, 'r', encoding='utf-8') as file:
    followerList = file.readlines()

print("팔로잉 목록 파일 이름 입력 ex) following.txt : ", end='')
followerFile = input()
with open(followerFile, 'r', encoding='utf-8') as file:
    followingList = file.readlines()

follower = extractIdList(followerList)
following = extractIdList(followingList)

print("집계된 팔로워 수 : ", len(follower))
print("집계된 팔로잉 수 : ", len(following))

cnt = 0
for id in following:
    if id not in follower:
        cnt += 1
        print(id)
print(cnt, "명이 당신을 맞팔 중이지 않음")