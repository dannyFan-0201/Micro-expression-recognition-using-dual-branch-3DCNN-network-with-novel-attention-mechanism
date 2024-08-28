import dlib
import numpy as np
import cv2

# dlib預測器
detector = dlib.get_frontal_face_detector()
# 讀取68點數據
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cv2讀取圖像
img = cv2.imread(r"C:\Users\a9401\Desktop\micro-expression-recognition-master\SMIC\test.JPG")
# 获取原始影像的尺寸
height, width = img.shape[:2]
# 设置目标尺寸
target_size = 224
# 计算填充的大小
top_pad = (target_size - height) // 2
bottom_pad = target_size - height - top_pad
left_pad = (target_size - width) // 2
right_pad = target_size - width - left_pad
img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
# 設置字體
font = cv2.FONT_HERSHEY_SIMPLEX

# 人臉數rects
rects = detector(img, 0)

# for i, rect in enumerate(rects):
#     # 獲取點矩陣68*2
#     landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
#     keypoints_to_cut = [3, 5, 13, 15]
#
#     # 找到人臉的範圍
#     min_x = np.min(landmarks[keypoints_to_cut, 0])
#     max_x = np.max(landmarks[keypoints_to_cut, 0])
#     min_y = np.min(landmarks[keypoints_to_cut, 1])
#     max_y = np.max(landmarks[keypoints_to_cut, 1])
#
#     # 切割出臉部圖像
#     face_img = img[min_y:max_y, min_x:max_x]
#
#     # 保存切割的臉部圖像，這里使用了每張臉的編號作為文件名
#     cv2.imwrite(f'face_{i}.jpg', face_img)
#
#     # 在原圖上畫出每個人臉的矩形框和關鍵點
#     cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
#     for point in landmarks:
#         pos = (point[0, 0], point[0, 1])
#         cv2.circle(img, pos, 1, (0, 0, 255), -1)
#
#     for idx, point in enumerate(landmarks):
#         # 68点的坐标
#         pos = (point[0, 0], point[0, 1])
#         # 利用cv2.circle给每个特征点画一个点，共68个
#         cv2.circle(img, pos, 1, (0, 0, 255), -1)
#         # 避免数字标签与68点重合，坐标微微移动
#         pos = list(pos)
#         pos[0] = pos[0] + 5
#         pos[1] = pos[1] + 5
#         pos = tuple(pos)
#         # 利用cv2.putText输出1-68的标签，不需要可以删除
#         cv2.putText(img, str(idx + 1), pos, font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

# # 顯示結果
# cv2.namedWindow("python_68_points", 2)
# cv2.imshow("python_68_points", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
for face in rects:
    landmarks = predictor(img, face)

    # 將關鍵點位置轉換為 NumPy 陣列
    landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])

    # 計算關鍵點之間的距離矩陣
    distance_matrix = np.linalg.norm(landmark_points[:, None, :] - landmark_points[None, :, :], axis=-1)

    # 根據距離矩陣計算鄰接矩陣，例如使用閾值來確定關係
    adjacency_matrix = (distance_matrix < 10).astype(int)  # 假設距離小於 50 的關鍵點之間有關係

    # 打印鄰接矩陣
    print("Shape of adjacency matrix:", adjacency_matrix.shape)
    print("Adjacency matrix:")
    print(adjacency_matrix)

