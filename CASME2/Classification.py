import pandas as pd
import shutil
import os

# 指定 Excel 檔案的路徑
excel_file_path = r"C:\Users\user\Desktop\micro-expression-recognition-master\CASME2-coding-20140508"

# 指定要讀取的工作表名稱
sheet_name = 'Sheet1'

# 讀取 Excel 檔案
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# 篩選 "emotion" 列中值為 "disgust" 的資料行
disgust_data = df[df['emotion'] == 'others']
# 重新命名使用前三列的資訊
disgust_data['New Name'] = disgust_data.apply(lambda row: f"{row['Subject']}_{row['Filename']}_{row['Onset']}", axis=1)

# 輸出符合條件的資料行
print(disgust_data)

# 指定源資料夾和目標資料夾路徑
source_folder = r'E:\CASMEdata\CAS(ME)3\part_A\Part_A_ME_clip\video'
target_folder = r'E:\CASMEdata\CAS(ME)3\part_A\video\others'
for index, row in disgust_data.iterrows():
    new_name = row['New Name']
    source_file_path = os.path.join(source_folder, f"{new_name}.mp4")  # 將".ext"替換為實際的文件擴展名
    target_file_path = os.path.join(target_folder, f"{new_name}.mp4")  # 替換".ext"為實際的文件擴展名
    try:
        shutil.copy(source_file_path, target_file_path)
    except FileNotFoundError:
        print(f"File not found for New Name: {new_name}. Skipping...")
        continue