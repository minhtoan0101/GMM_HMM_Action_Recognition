import cv2

VIDEO_PATH = r"D:\GMM_HMM_Action_Recognition\dataset\PushUps\v_PushUps_g01_c01.avi"

print("Đang kiểm tra:")
print(VIDEO_PATH)
print()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Không thể mở video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = frame_count / fps if fps > 0 else 0

print("Video mở thành công!")
print(f"FPS        : {fps}")
print(f"Frames     : {frame_count}")
print(f"Resolution : {width} x {height}")
print(f"Duration   : {duration:.2f} seconds")

cap.release()