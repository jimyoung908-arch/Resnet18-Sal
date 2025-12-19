import cv2
import numpy as np

def cv_imread(file_path, type=0):
    """
    支持中文路径的图片读取函数
    type: 0 for Gray (Mask), 1 for Color (Image)
    """
    try:
        # 使用 numpy 读取文件流，再解码，避开路径字符集问题
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)

        if cv_img is None:
            raise FileNotFoundError(f"无法读取文件: {file_path}")

        if type == 0: # 预期读取灰度图
            if len(cv_img.shape) == 3:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else: # 预期读取彩色图
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

        return cv_img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None