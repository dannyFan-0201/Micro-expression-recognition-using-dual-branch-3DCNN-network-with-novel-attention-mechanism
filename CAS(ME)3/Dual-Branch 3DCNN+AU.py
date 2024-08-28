import os
import cv2
import imageio
import dlib
import re
import pandas as pd
import numpy as np
from keras.layers.core import Dropout, Activation
from keras.layers import LSTM, TimeDistributed, GRU
from keras.models import Model
from keras.layers import Input, Conv3D, LeakyReLU, concatenate, Flatten, Dense, MaxPooling1D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import Callback
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from attention import CBAMModule, MultiScaleMultiHeadAttention, SingleHeadAttention, MultiHeadAttention
from keras.regularizers import l1, l2
from keras.utils import np_utils
from skimage.metrics import structural_similarity as ssim



class ClassLossCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_val, x_val2, y_val = self.validation_data

        # confusion_matrix
        predictions = self.model.predict((x_val, x_val2))
        y_pred = np.argmax(predictions, axis=-1)
        y_true = np.argmax(y_val, axis=-1)
        y_pred = y_pred[:, 0]
        cfm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cfm)

        # 計算混淆矩陣
        cfm = tf.math.confusion_matrix(y_true, y_pred)

        # acc
        acc_score = acc_score = tf.round(tf.reduce_sum(tf.linalg.diag_part(cfm)) / tf.reduce_sum(cfm) * 1e7) / 1e7

        # UAR
        true_positives = tf.linalg.diag_part(cfm)
        true_positives = tf.cast(true_positives, tf.float32)
        actual_positives = tf.reduce_sum(cfm, axis=1)
        actual_positives_safe = tf.where(tf.equal(actual_positives, 0), tf.constant(1e-7, dtype=tf.float32), tf.cast(actual_positives, tf.float32))
        recall_per_class = true_positives / actual_positives_safe
        uar = tf.cond(tf.equal(acc_score, 1), lambda: acc_score, lambda: tf.reduce_mean(recall_per_class))

        # UF1
        precision_per_class = true_positives / tf.cast(tf.reduce_sum(cfm, axis=0), tf.float32)
        precision_per_class_safe = tf.where(tf.math.is_finite(precision_per_class), precision_per_class,tf.constant(1e-7, dtype=tf.float32))
        f1_per_class = 2 * (precision_per_class_safe * recall_per_class) / (precision_per_class_safe + recall_per_class + tf.constant(1e-7, dtype=tf.float32))
        uf1 = tf.cond(tf.equal(acc_score, 1), lambda: acc_score, lambda: tf.reduce_mean(f1_per_class))

        print("UAR:", uar.numpy())
        print("UF1:", uf1.numpy())
        uar_list.append(uar.numpy())
        uf1_list.append(uf1.numpy())
        acc_list.append(acc_score.numpy())
        uf1_value = max(uf1_list)
        if uf1_value == 1:
            self.model.stop_training = True

def weighted_categorical_crossentropy(class_weights):
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        weighted_losses = -tf.reduce_sum(class_weights * y_true * tf.math.log(tf.maximum(y_pred, 1e-15)), axis=-1)

        return weighted_losses
    return loss



def calculate_frame_difference(frame1, frame2):
    # 計算兩幀之間的絕對差異
    diff = cv2.absdiff(frame1, frame2)
    # 將差異轉換為灰度圖像
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 計算灰度圖像中所有像素的總和
    return np.sum(gray_diff)



def process_video_frames(video_path, image_rows, image_columns, image_depth, output_folder, AU_numbers_dict):
    # 初始化人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    frames = []
    local_frames = []
    frame_diffs = []
    loadedvideo = imageio.get_reader(video_path, 'ffmpeg')
    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    # 根据字典查找对应的值
    AU_value = AU_numbers_dict.get(video_name, 0)
    total_frames = loadedvideo.count_frames()
    # print(video_name, total_frames)
    # 讀取所有幀，計算差異並存儲
    prev_frame = None
    for frame_idx in range(0, total_frames):
        image = loadedvideo.get_data(frame_idx)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if prev_frame is not None:
            diff = calculate_frame_difference(prev_frame, image)
            frame_diffs.append(diff)
        prev_frame = image
    # 选取变化最大的**帧的索引
    top_indices = np.argsort(frame_diffs)[-30:]
    # 按索引排序
    frame_indices = sorted(top_indices)
    frame_counter = 0

    for frame in frame_indices:
        # 获取特定帧
        image = loadedvideo.get_data(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        frames.append(imageresize)
        faces = detector(image)
        face_img_resize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)

        for face in faces:
            landmarks = np.matrix([[p.x, p.y] for p in predictor(image, face).parts()])
            if isinstance(AU_value, list):
                if any(au in [4, 5, 7, 43] for au in AU_value):
                    face_lm = [19, 24, 15, 1]  # eyes
                elif any(au in [10, 12, 14, 15, 17] for au in AU_value):
                    face_lm = [3, 5, 11, 13]  # mouth
                elif any(au in [1, 2] for au in AU_value):
                    face_lm = [17, 19, 24, 26]  # eyebrow
                elif any(au in [6, 9] for au in AU_value):
                    face_lm = [1, 29, 15, 3, 13]  # cheek
                else:
                    face_lm = list(range(68))
            else:
                face_lm = list(range(68))
            min_x = np.min(landmarks[face_lm, 0])
            max_x = np.max(landmarks[face_lm, 0])
            min_y = np.min(landmarks[face_lm, 1])
            max_y = np.max(landmarks[face_lm, 1])
            face_img = image[min_y:max_y, min_x:max_x]

            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                face_img_resize = cv2.resize(face_img, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            else:
                face_img_resize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        frame_filename = f"{frame_counter}.jpg"
        frame_filepath = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_filepath, face_img_resize)
        local_frames.append(face_img_resize)
        frame_counter += 1

    frames = np.asarray(frames)
    local_frames = np.asarray(local_frames)
    return frames.transpose(0, 1, 2, 3), local_frames.transpose(0, 1, 2, 3)


def process_videos_in_path(path, label, image_rows, image_columns, image_depth, output_folder, AU_numbers_dict):
    training_list = []
    training_list2 = []
    video_list = os.listdir(path)
    for video in video_list:
        video_path = os.path.join(path, video)
        videoarray, local_videoarray = process_video_frames(video_path, image_rows, image_columns, image_depth, output_folder, AU_numbers_dict)
        training_list.append(videoarray)
        training_list2.append(local_videoarray)
    return training_list, training_list2, video_list

def extract_AU_numbers(AUdata):
    # 确保 AUdata 是字符串类型
    if not isinstance(AUdata, str):
        raise ValueError("AUdata should be a string")
    # 使用正则表达式匹配所有的数字部分
    numbers = re.findall(r'\d+', AUdata)
    # 将匹配到的数字部分转换为整数并存储在列表中
    AU_numbers = [int(number) for number in numbers]
    return AU_numbers

# # 创建文件夹
# output_folder = "training_frames"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# frame_counter = 0

# 参数设置
image_rows, image_columns, image_depth = 128, 128, 30
output_folder = r'C:\Users\user\Desktop\micro-expression-recognition-master\CAS(ME)3\training_frames'
negativepath = r'C:\Users\user\Desktop\micro-expression-recognition-master\CAS(ME)3\data\negative'
positivepath = r'C:\Users\user\Desktop\micro-expression-recognition-master\CAS(ME)3\data\positive/'
surprisepath = r'C:\Users\user\Desktop\micro-expression-recognition-master\CAS(ME)3\data\surprise/'
# otherspath = 'C:/Users/user/Desktop/micro-expression-recognition-master/CASME/data(CAS(ME)2)/others/'
# 指定 Excel 檔案的路徑
excel_file_path = r"C:\Users\user\Desktop\micro-expression-recognition-master\cas(me)3_part_A_ME_label_JpgIndex_v1.xls"
df = pd.read_excel(excel_file_path, sheet_name='Sheet1')
# 将 Action Units 数据转换为字符串
df["Action Units"] = df["Action Units"].astype(str)
AU_CODE = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 43]
# 提取 Action Units 數字並過濾掉不在 AU_CODE 中的數字
def filter_AU_numbers(AUdata, AU_CODE):
    numbers = extract_AU_numbers(AUdata)
    filtered_numbers = [number for number in numbers if number in AU_CODE]
    return filtered_numbers

AU_list = [filter_AU_numbers(AUdata, AU_CODE) for AUdata in df.loc[:, "Action Units"]]
AU_numbers_dict = {f"{subject}_{filename}_{onset}": au for (subject, filename, onset), au in zip(zip(df['Subject'], df['Filename'], df['Onset']), AU_list)}
# print(name_dict)
# AU_numbers_dict = dict(zip(name_dict, AU_list))
print("Extracted AU numbers list:", AU_numbers_dict)


# 处理每个类别的视频
neg_videos, neg_videos_local, neg_list = process_videos_in_path(negativepath, 0, image_rows, image_columns, image_depth, output_folder, AU_numbers_dict)
pos_videos, pos_videos_local, pos_list = process_videos_in_path(positivepath, 1, image_rows, image_columns, image_depth, output_folder, AU_numbers_dict)
sur_videos, sur_videos_local, sur_list = process_videos_in_path(surprisepath, 2, image_rows, image_columns, image_depth, output_folder, AU_numbers_dict)
# oth_videos, oth_videos_local, oth_list = process_videos_in_path(otherspath, 3, image_rows, image_columns, image_depth, output_folder, AU_numbers_dict)

# 合并所有视频
training_list = neg_videos + pos_videos + sur_videos
training_list2 = neg_videos_local + pos_videos_local + sur_videos_local
video_list = neg_list + pos_list + sur_list
print("Length of training_list:", len(training_list))
print("Length of training_list2:", len(training_list2))


# 转换为numpy数组
training_list = np.asarray(training_list)
training_list2 = np.asarray(training_list2)
trainingsamples = len(training_list)

# 创建训练标签
traininglabels = np.zeros((trainingsamples,), dtype=int)
traininglabels[0:len(neg_videos)] = 0
traininglabels[len(neg_videos):len(neg_videos) + len(pos_videos)] = 1
traininglabels[len(neg_videos) + len(pos_videos):] = 2
# traininglabels[len(neg_videos) + len(pos_videos) + len(sur_videos):] = 3
traininglabels = np_utils.to_categorical(traininglabels, 3)

# 创建训练数据
training_data = [training_list, traininglabels]
training_data2 = [training_list2, traininglabels]
trainingframes, traininglabels = training_data
trainingframes2, traininglabels = training_data2

training_set = np.zeros((trainingsamples, image_depth, image_rows, image_columns, 3))
training_set2 = np.zeros((trainingsamples, image_depth, image_rows, image_columns, 3))

for h in range(trainingsamples):
    training_set[h] = trainingframes[h]
    training_set2[h] = trainingframes2[h]

training_set = training_set.astype('float32')
training_set -= np.mean(training_set)
training_set /= np.max(training_set)

training_set2 = training_set2.astype('float32')
training_set2 -= np.mean(training_set2)
training_set2 /= np.max(training_set2)

# print("Training set shape:", training_set.shape)
# print("Training set2 shape:", training_set2.shape)
# print("Labels shape:", traininglabels.shape)

# Load pre-trained weights
# model.load_weights(r'C:\Users\user\Desktop\micro-expression-recognition-master\CASME-SQUARE\weights_microexpstcnn\weights-improvement-86-0.67.hdf5')


# 讀取 Excel 檔案
cleaned_video_list = [video.replace('.mp4', '') for video in video_list]
label_dict = dict(zip(AU_numbers_dict, df['Subject']))
label_list = [label_dict.get(video, 'unknown') for video in cleaned_video_list]
total_l = len(label_list)
total_v = len(video_list)
print(cleaned_video_list)
print(label_list)
print(total_v)
print(total_l)

# 定義你的訓練資料和標籤
X = training_set
X2 = training_set2
y = traininglabels

# 定義每個樣本所屬的主題或子集，例如，這是一個主題列表
subjects = label_list

# 評估指標
Uar = []
Uf1 = []
acc = []
i = 1


# 使用LeaveOneGroupOut來進行LOSO交叉驗證
logo = LeaveOneGroupOut()
for train_index, test_index in logo.split(X, y, groups=subjects):
    print("\n "+ f"##### Subject: {i} #####\n")
    train_images, validation_images = X[train_index], X[test_index]
    train_labels, validation_labels = y[train_index], y[test_index]
    train_images2, validation_images2 = X2[train_index], X2[test_index]

    test_files = [cleaned_video_list[i] for i in test_index]
    print(f"Testing with files: {', '.join(test_files)}")

    #  Model  epoch:200
    # 初始輸入形狀
    input_shape = (image_depth, image_rows, image_columns, 3)
    input_layer = Input(shape=input_shape)
    input_shape2 = (image_depth, image_rows, image_columns, 3)
    input_layer2 = Input(shape=input_shape2)
    # 第一個3D CNN分支
    CNN1 = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='he_normal')(input_layer)
    CNN1 = LeakyReLU(alpha=0.2)(CNN1)
    CNN1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(CNN1)

    # 第二個3D CNN分支
    CNN2 = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer='he_normal')(input_layer2)
    CNN2 = LeakyReLU(alpha=0.2)(CNN2)
    CNN2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(CNN2)

    # 連接兩個分支
    merged = concatenate([CNN1, CNN2])
    CNN = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer='he_normal')(merged)
    CNN = LeakyReLU(alpha=0.2)(CNN)
    CNN = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(CNN)
    # 將模型組合為一個完整的模型
    output_layer = CBAMModule(channels=64, reduction=8)(CNN)
    output_layer = TimeDistributed(Flatten())(output_layer)
    output_layer = Dropout(0.5)(output_layer)
    output_layer = TimeDistributed(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)))(output_layer)
    output_layer = LeakyReLU(alpha=0.2)(output_layer)
    output_layer = Dropout(0.5)(output_layer)
    output_layer = TimeDistributed(Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)))(output_layer)
    output_layer = LeakyReLU(alpha=0.2)(output_layer)
    output_layer = GRU(512, return_sequences=True)(output_layer)
    # 在原始的 GRU 層上添加自注意力机制
    d_model = 512
    num_heads = 8
    scale_factors = [2, 4, 8, 16, 32, 64, 128, 256]
    multi_scale_attention = MultiScaleMultiHeadAttention(d_model, num_heads, scale_factors)
    attention_out = multi_scale_attention(output_layer, output_layer, output_layer)

    # 合併 GRU 输出和自注意力输出
    combined_output = Dense(128, kernel_initializer='he_normal')(attention_out)
    combined_output = LeakyReLU(alpha=0.2)(combined_output)
    combined_output = Dropout(0.3)(combined_output)
    combined_output = Dense(64, kernel_initializer='he_normal')(combined_output)
    combined_output = LeakyReLU(alpha=0.2)(combined_output)
    combined_output = Dropout(0.3)(combined_output)
    output_layer_final = Dense(3, activation='softmax')(combined_output)
    # 建立整合後的模型
    model = Model(inputs=[input_layer, input_layer2], outputs=output_layer_final)
    class_weights = [0.3, 0.35, 0.35]
    loss = weighted_categorical_crossentropy(class_weights)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # 評估指標
    uar_list = []
    uf1_list = []
    acc_list = []

    # 定義衰減函數
    def lr_scheduler(epoch, lr):
        if epoch % 20 == 0 and epoch != 0:
            return lr * 0.9
        else:
            return lr

    initial_weights = model.get_weights()
    model.set_weights(initial_weights)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    filepath = "weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    # Training the model
    tensorboard_callback = TensorBoard(log_dir=r"C:\Users\user\Desktop\micro-expression-recognition-master\logs")
    callbacks_list.append(tensorboard_callback)
    callbacks_list.append(ClassLossCallback((validation_images, validation_images2, validation_labels)))
    callbacks_list.append(lr_callback)
    train_labels = np.repeat(train_labels[:, np.newaxis, :], 3, axis=1)
    validation_labels = np.repeat(validation_labels[:, np.newaxis, :], 3, axis=1)
    # hist = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), callbacks=callbacks_list, batch_size=8, epochs=200, shuffle=True)
    hist = model.fit((train_images, train_images2), train_labels,validation_data=((validation_images, validation_images2), validation_labels),
                     callbacks=callbacks_list, batch_size=8, epochs=100, shuffle=True)

    best_uf1_index = uf1_list.index(max(uf1_list))
    best_uar_for_best_uf1 = uar_list[best_uf1_index]
    print("Best UAR:", best_uar_for_best_uf1)
    print("Best UF1:", uf1_list[best_uf1_index])
    print("Best ACC:", max(acc_list))
    Uar.append(best_uar_for_best_uf1)
    Uf1.append(uf1_list[best_uf1_index])
    acc.append(max(acc_list))
    i += 1
    tf.keras.backend.clear_session()


#計算最終指標分數
for index, (value1, value2, value3) in enumerate(zip(Uar, Uf1, acc)):
    formatted_value1 = "{:.4f}".format(value1).rstrip('0').rstrip('.')
    formatted_value2 = "{:.4f}".format(value2).rstrip('0').rstrip('.')
    formatted_value3 = "{:.4f}".format(value3).rstrip('0').rstrip('.')
    print(f" {index + 1}: UAR={formatted_value1}, UF1={formatted_value2}, ACC={formatted_value3}")

# print(Uar)
# print(Uf1)

average_Uar = sum(Uar) / len(Uar)
average_Uf1 = sum(Uf1) / len(Uf1)
average_acc = sum(acc) / len(acc)
print("Average UF1:", average_Uf1)
print("Average Uar:", average_Uar)
print("Average ACC:", average_acc)






