import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 超参数设置
BATCH_SIZE = 5
num_epochs = 50
LEARNING_RATE = 0.001  # 新增学习率参数
PATIENCE = 10  # 早停策略的耐心值

def resize_image_with_padding(image, target_size=(1024, 1024)):
    height, width = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return padded_image

def load_data():
    train_dir = 'train'
    all_images = []
    all_states = []
    all_labels = []

    for state in os.listdir(train_dir):
        state_dir = os.path.join(train_dir, state)
        if os.path.isdir(state_dir):
            image_files = [f for f in os.listdir(state_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in image_files:
                img_path = os.path.join(state_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = resize_image_with_padding(img, target_size=(1024, 1024))
                    img = img / 255.0  # 归一化到 [0, 1] 范围
                    all_images.append(img)

                    txt_file = os.path.splitext(img_file)[0] + '.txt'
                    txt_path = os.path.join(state_dir, txt_file)
                    try:
                        label = np.loadtxt(txt_path)
                        all_labels.append(label - 1)
                    except Exception as e:
                        print(f"Error loading label from {txt_path}: {e}")
                        continue

                    all_states.append(int(os.path.splitext(img_file)[0]) - 1)

    # 填充坐标标签
    max_length = max([len(label) for label in all_labels])
    padded_labels = []
    for label in all_labels:
        padded = pad_sequences([label], maxlen=max_length, padding='post', value=0)
        padded_labels.append(padded[0])

    all_labels = np.array(padded_labels)
    all_states = np.array(all_states)
    all_images = np.array(all_images)

    return all_images, all_states, all_labels

def create_model(input_shape):
    base_model = models.Sequential([
        # 第一层卷积
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 第二层卷积
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 第三层卷积
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 全局平均池化，减少参数量
        layers.GlobalAveragePooling2D(),

        # 连接到全连接层
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
    ])

    # 多任务输出：人脸状态和坐标预测
    state_output = layers.Dense(4, activation='softmax', name='state_output')(base_model.output)
    point_output = layers.Dense(8, activation='linear', name='point_output')(base_model.output)

    model = models.Model(inputs=base_model.input, outputs=[state_output, point_output])
    return model

def train_model(model, images, states, points):
    points = points.reshape(-1, 8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)  # 使用自定义学习率

    model.compile(
        optimizer=optimizer,
        loss={
            'state_output': 'sparse_categorical_crossentropy',
            'point_output': 'mse'
        },
        metrics={
            'state_output': 'accuracy',
            'point_output': 'mae'
        }
    )

    # 早停策略
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(
        images,
        {'state_output': states, 'point_output': points},
        epochs=num_epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    return model, history

def test_model(model, test_image):
    test_image = np.expand_dims(test_image, axis=0)
    state_pred, point_pred = model.predict(test_image)
    state_index = np.argmax(state_pred)
    points = point_pred.reshape(-1, 2)

    return state_index, points

def mark_points(image, points):
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    return image

if __name__ == "__main__":
    all_images, all_states, all_labels = load_data()

    # 划分训练集和测试集
    images_train, images_test, states_train, states_test, labels_train, labels_test = train_test_split(
        all_images, all_states, all_labels, test_size=0.2, random_state=42
    )

    model = create_model(input_shape=(1024, 1024, 3))
    model, history = train_model(model, images_train, states_train, labels_train)

    if history is not None:
        # 以下是显示训练测试集和训练集的loss以及accurancy折线图
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 创建 2x2 的子图布局

        # 绘制状态损失曲线
        axs[0, 0].plot(history.history['state_output_loss'], label='state_output_loss')
        axs[0, 0].plot(history.history['val_state_output_loss'], label='val_state_output_loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        # 绘制点位损失曲线
        axs[0, 1].plot(history.history['point_output_loss'], label='point_output_loss')
        axs[0, 1].plot(history.history['val_point_output_loss'], label='val_point_output_loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()

        # 绘制点位准确率曲线
        axs[1, 0].plot(history.history['state_output_accuracy'], label='state_output_accuracy')
        axs[1, 0].plot(history.history['val_state_output_accuracy'], label='val_state_output_accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].legend()

        # 绘制点位MAE曲线
        axs[1, 1].plot(history.history['point_output_mae'], label='point_output_mae')
        axs[1, 1].plot(history.history['val_point_output_mae'], label='val_point_output_mae')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('MAE')
        axs[1, 1].legend()

        plt.tight_layout()  # 调整子图布局，避免重叠
        plt.show()

    if model is not None:
        test_image = cv2.imread(r"C:\Users\40171\Desktop\488bac009ca025b9be39fba336d261e.jpg")
        test_image = resize_image_with_padding(test_image, target_size=(1024, 1024))
        test_image = resize_image_with_padding(test_image, target_size=(1024, 1024))
        test_image = test_image / 255.0  # 归一化到 [0, 1] 范围
        state_index, points = test_model(model, test_image)

        state_mapping = {0: '正面', 1: '右侧面', 2: '左侧面', 3: '后脑'}
        print(f"预测的人脸状态: {state_mapping[state_index]}")

        model.save('acupuncture_point_model')

        marked_image = mark_points((test_image * 255).astype(np.uint8), points)
        cv2.imshow("Marked Points", marked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
