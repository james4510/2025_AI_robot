import tensorflow as tf
import numpy as np

# 훈련 데이터 생성
x_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# 가중치와 편향 변수 정의
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# 모델 정의 (선형 함수)
def linear_regression(x):
    return w*x+b

# 손실 함수 정의 (평균 제곱 오차)
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y-y_pred))

# 최적화 알고리즘 선택 (경사 하강법)
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 훈련 함수 정의
def train_step(x, y):
    with tf.GradientTape() as tape:
        # 순전파 계산
        y_pred = linear_regression(x)
        # 손실 계산
        loss_value = loss(y, y_pred)

    # 가중치와 편향에 대한 기울기 계산
    gradients = tape.gradient(loss_value, [w, b])

    # 최적화 알고리즘을 사용하여 가중치와 편향 업데이트
    optimizer.apply_gradients(zip(gradients, [w, b]))

# 훈련
epochs = 1000
for epoch in range(epochs):
    # 한 번의 훈련 스텝 수행
    train_step(x_train, y_train)

# 훈련된 모델의 예측 결과 확인
x_test = np.array([6, 7, 8, 9, 10], dtype=np.float32)
y_pred = linear_regression(x_test)
print("예측 결과: ", y_pred.numpy())