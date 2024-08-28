import cv2
import os

# 输入影片所在的文件夹
input_folder = r'C:\Users\a9401\Desktop\micro-expression-recognition-master\data\1'
# 输出影片所在的文件夹
output_folder = r'C:\Users\a9401\Desktop\micro-expression-recognition-master\data\negative_aug'

# 获取文件夹中的所有影片文件
video_files = [f for f in os.listdir(input_folder)]

for video_file in video_files:
    # 组合输入和输出影片的完整路径
    input_video = os.path.join(input_folder, video_file)
    output_video = os.path.join(output_folder, f'flipped_{video_file}')

    # 读取输入影片
    cap = cv2.VideoCapture(input_video)

    # 获取原始影片的帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 获取原始影片的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置影片写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 水平翻转帧
        flipped_frame = cv2.flip(frame, 1)

        # 逆时针旋转5度
        # rotation_matrix = cv2.getRotationMatrix2D((frame_width / 2, frame_height / 2), -5, 1)
        # rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))

        # 将翻转后的帧写入输出影片
        out.write(flipped_frame)

    # 释放资源
    cap.release()
    out.release()

    print(f"影片 {video_file} 已经逆時針旋轉并保存为 {output_video}")
