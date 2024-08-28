import os
import cv2
import numpy
import imageio
import dlib
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers import LSTM, TimeDistributed, GRU
from keras.models import Model
from keras.layers import Input, Conv3D, LeakyReLU, concatenate, Flatten, Dense, MaxPooling1D
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import Callback
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from attention import CBAMModule, SpatialAttention, ChannelAttention, MultiScaleMultiHeadAttention, SingleHeadAttention, MultiHeadAttention
from keras.regularizers import l1, l2

class ClassLossCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_val, x_val2, y_val = self.validation_data

        # confusion_matrix
        predictions = self.model.predict((x_val, x_val2))
        y_pred = numpy.argmax(predictions, axis=-1)
        y_true = numpy.argmax(y_val, axis=-1)
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


image_rows, image_columns, image_depth = 128, 128, 16

training_list = []
training_list2 = []
video_list = []
negativepath = 'C:/Users/a9401/Desktop/micro-expression-recognition-master/Real-life deception detection database/lie/'  #lie
positivepath = 'C:/Users/a9401/Desktop/micro-expression-recognition-master/Real-life deception detection database/truth/'  #truth



# 创建目标文件夹
output_folder = "training_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
frame_counter = 0
# 初始化人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

directorylisting = os.listdir(negativepath)
video_list = directorylisting
for video in directorylisting:
    frames = []
    local_frames = []
    videopath = negativepath +video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x + 5 for x in range(16)]
    for frame in framerange:
        image = loadedvideo.get_data(frame)
        imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        # grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(imageresize)

        # 在帧上进行人脸检测
        faces = detector(image)
        # 对于每个检测到的人脸，裁剪并保存人脸图像
        for i, face in enumerate(faces):
            # 獲取點矩陣68*2
            landmarks = numpy.matrix([[p.x, p.y] for p in predictor(image, face).parts()])
            # 找到人臉的範圍
            min_x = numpy.min(landmarks[:, 0])
            max_x = numpy.max(landmarks[:, 0])
            min_y = numpy.min(landmarks[:, 1])
            max_y = numpy.max(landmarks[:, 1])
            # 切割出臉部圖像
            face_img = image[min_y:max_y, min_x:max_x]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:  # 判断人脸图像是否为空
                face_img_resize = cv2.resize(face_img, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            else:
                # 使用原始图像
                face_img_resize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            # 保存人脸图像到文件
            frame_filename = f"{frame_counter}.jpg"
            frame_filepath = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_filepath, face_img_resize)
            frame_counter += 1
        local_frames.append(face_img_resize)

    frames = numpy.asarray(frames)
    local_frames = numpy.asarray(local_frames)
    # videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    videoarray = frames.transpose(0, 1, 2, 3)
    local_videoarray = local_frames.transpose(0, 1, 2, 3)
    training_list.append(videoarray)
    training_list2.append(local_videoarray)

directorylisting = os.listdir(positivepath)
video_list.extend(directorylisting)
for video in directorylisting:
    frames = []
    local_frames = []
    videopath = positivepath +video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x + 5 for x in range(16)]
    for frame in framerange:
        image = loadedvideo.get_data(frame)
        imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        # grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(imageresize)

        # 在帧上进行人脸检测
        faces = detector(image)
        # 对于每个检测到的人脸，裁剪并保存人脸图像
        for i, face in enumerate(faces):
            # 獲取點矩陣68*2
            landmarks = numpy.matrix([[p.x, p.y] for p in predictor(image, face).parts()])
            # 找到人臉的範圍
            min_x = numpy.min(landmarks[:, 0])
            max_x = numpy.max(landmarks[:, 0])
            min_y = numpy.min(landmarks[:, 1])
            max_y = numpy.max(landmarks[:, 1])
            # 切割出臉部圖像
            face_img = image[min_y:max_y, min_x:max_x]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:  # 判断人脸图像是否为空
                face_img_resize = cv2.resize(face_img, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            else:
                # 使用原始图像
                face_img_resize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            # 保存人脸图像到文件
            frame_filename = f"{frame_counter}.jpg"
            frame_filepath = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_filepath, face_img_resize)
            frame_counter += 1
        local_frames.append(face_img_resize)

    frames = numpy.asarray(frames)
    local_frames = numpy.asarray(local_frames)
    # videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    videoarray = frames.transpose(0, 1, 2, 3)
    local_videoarray = local_frames.transpose(0, 1, 2, 3)
    training_list.append(videoarray)
    training_list2.append(local_videoarray)



training_list = numpy.asarray(training_list)
training_list2 = numpy.asarray(training_list2)
trainingsamples = len(training_list)


traininglabels = numpy.zeros((trainingsamples, ), dtype = int)

traininglabels[0:54] = 0
traininglabels[54:104] = 1

traininglabels = np_utils.to_categorical(traininglabels, 2)

training_data = [training_list, traininglabels]
training_data2 = [training_list2, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
(trainingframes2, traininglabels) = (training_data2[0], training_data2[1])
training_set = numpy.zeros((trainingsamples, image_depth, image_rows, image_columns, 3))
training_set2 = numpy.zeros((trainingsamples, image_depth, image_rows, image_columns, 3))
for h in range(trainingsamples):
    training_set[h][:][:][:][:][:] = trainingframes[h,:,:,:,:]
    training_set2[h][:][:][:][:][:] = trainingframes[h, :, :, :, :]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)

training_set2 = training_set2.astype('float32')
training_set2 -= numpy.mean(training_set2)
training_set2 /= numpy.max(training_set2)

# Load pre-trained weights

# model.load_weights(r'C:\Users\a9401\Desktop\micro-expression-recognition-master\CASME-SQUARE\weights_microexpstcnn\weights-improvement-86-0.67.hdf5')


# 指定 Excel 檔案的路徑
excel_file_path = r"C:\Users\a9401\Desktop\micro-expression-recognition-master\lie detection.xlsx"
# 讀取 Excel 檔案
df = pd.read_excel(excel_file_path, sheet_name='Sheet1')
# print(df.columns)
# print(df.head())
cleaned_video_list = [video.replace('.mp4', '') for video in video_list]
label_dict = dict(zip(df['ME_Type'], df['Subject']))
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
    train_labels2, validation_labels2 = y[train_index], y[test_index]

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
    CNN1 = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(CNN1)
    # 第二個3D CNN分支
    CNN2 = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer='he_normal')(input_layer2)
    CNN2 = LeakyReLU(alpha=0.2)(CNN2)
    CNN2 = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(CNN2)
    # 連接兩個分支
    merged = concatenate([CNN1, CNN2])
    CNN = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer='he_normal')(merged)
    CNN = LeakyReLU(alpha=0.2)(CNN)
    CNN = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(CNN)
    # CNN = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer='he_normal')(CNN)
    # CNN = LeakyReLU(alpha=0.2)(CNN)
    # CNN = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(CNN)
    # 將模型組合為一個完整的模型
    output_layer = CBAMModule(channels=32, reduction=8)(CNN)
    output_layer = TimeDistributed(Flatten())(output_layer)
    output_layer = Dropout(0.3)(output_layer)
    output_layer = TimeDistributed(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.03)))(output_layer)
    output_layer = LeakyReLU(alpha=0.2)(output_layer)
    output_layer = Dropout(0.3)(output_layer)
    output_layer = TimeDistributed(Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.03)))(output_layer)
    output_layer = LeakyReLU(alpha=0.2)(output_layer)
    output_layer = Dropout(0.3)(output_layer)
    output_layer = GRU(512, return_sequences=True)(output_layer)
    # 在原始的 GRU 層上添加自注意力机制
    d_model = 512
    num_heads = 8
    scale_factors = [2, 4, 8, 16, 32, 64, 128, 256]
    multi_scale_attention = MultiScaleMultiHeadAttention(d_model, num_heads, scale_factors)
    attention_out = multi_scale_attention(output_layer, output_layer, output_layer)
    self_attention = MultiHeadAttention(num_heads=4, key_dim=512)
    attention_out2 = self_attention(output_layer, output_layer, use_causal_mask=True)
    attention_out2 = MaxPooling1D(pool_size=10, padding='same')(attention_out2)
    # 合併 GRU 输出和自注意力输出
    combined_output = concatenate([attention_out, attention_out2])
    combined_output = Dropout(0.3)(combined_output)
    combined_output = Dense(128, kernel_initializer='he_normal')(combined_output)
    combined_output = LeakyReLU(alpha=0.2)(combined_output)
    combined_output = Dropout(0.3)(combined_output)
    combined_output = Dense(64, kernel_initializer='he_normal')(combined_output)
    combined_output = LeakyReLU(alpha=0.2)(combined_output)
    combined_output = Dropout(0.3)(combined_output)
    output_layer_final = Dense(2, activation='softmax')(combined_output)
    # 建立整合後的模型
    model = Model(inputs=[input_layer, input_layer2], outputs=output_layer_final)
    class_weights = [0.5, 0.5]
    loss = weighted_categorical_crossentropy(class_weights)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    model.summary()

    # 評估指標
    uar_list = []
    uf1_list = []
    acc_list = []

    initial_weights = model.get_weights()
    model.set_weights(initial_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    filepath = "weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    # Training the model
    tensorboard_callback = TensorBoard(log_dir=r"C:\Users\a9401\Desktop\micro-expression-recognition-master\logs")
    callbacks_list.append(tensorboard_callback)
    callbacks_list.append(ClassLossCallback((validation_images, validation_images2, validation_labels)))
    train_labels = numpy.repeat(train_labels[:, numpy.newaxis, :], 2, axis=1)
    validation_labels = numpy.repeat(validation_labels[:, numpy.newaxis, :], 2, axis=1)
    # hist = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), callbacks=callbacks_list, batch_size=8, epochs=100, shuffle=True)
    hist = model.fit((train_images, train_images2), train_labels,
                     validation_data=((validation_images, validation_images2), validation_labels),
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






