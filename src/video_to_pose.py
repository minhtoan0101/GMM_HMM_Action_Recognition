import cv2
import mediapipe as mp
from pathlib import Path

# PATH
# Thư mục gốc của project:
# D:\GMM_HMM_Action_Recognition
PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "JumpingJack"
    / "v_JumpingJack_g01_c01.avi"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "pose_landmarker.task"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "output"
    / "pose_result.mp4"
)

# CHECK FILE

print("================================")
print("PATH INFORMATION")
print("================================")

print(f"Project : {PROJECT_DIR}")
print(f"Video   : {VIDEO_PATH}")
print(f"Model   : {MODEL_PATH}")
print(f"Output  : {OUTPUT_PATH}")
print()


if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy video:\n{VIDEO_PATH}"
    )


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy model:\n{MODEL_PATH}"
    )

# MEDIA PIPE

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

# POSE CONNECTIONS

# MediaPipe Pose có 33 landmarks.
#
# Các cặp dưới đây xác định những điểm nào được nối
# với nhau để tạo thành skeleton người.

POSE_CONNECTIONS = [

    # Vai
    (11, 12),
    # Tay trái
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    # Tay phải
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    # Thân người
    (11, 23),
    (12, 24),
    (23, 24),
    # Chân trái
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    # Chân phải
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32)
]

# OPEN VIDEO

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        f"Không thể mở video:\n{VIDEO_PATH}"
    )

# VIDEO INFORMATION

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("================================")
print("VIDEO INFORMATION")
print("================================")

print(f"FPS       : {fps}")
print(f"Frames    : {frame_count}")
print(f"Resolution: {width} x {height}")

print()

# OUTPUT VIDEO

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        f"Không thể tạo video output:\n{OUTPUT_PATH}"
    )

# POSE OPTIONS

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(
            MODEL_PATH
        )
    ),

    running_mode=VisionRunningMode.VIDEO,

    # Chỉ nhận diện 1 người
    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)

# PROCESS VIDEO

print("================================")
print("START POSE DETECTION")
print("================================")
print()

frame_index = 0


with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # Đọc frame
        success, frame = cap.read()


        if not success:
            break

        # OpenCV sử dụng BGR
        # MediaPipe sử dụng RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Tạo MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp
        #
        # MediaPipe VIDEO mode yêu cầu timestamp
        # tăng dần theo thời gian.

        timestamp_ms = int(
            frame_index * 1000 / fps
        )

        # Detect Pose

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )

        # DRAW POSE

        if result.pose_landmarks:

            # Có thể có nhiều người,
            # nhưng hiện tại num_poses = 1.

            for pose_landmarks in result.pose_landmarks:

                # DRAW LANDMARKS

                for landmark in pose_landmarks:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    # Chỉ vẽ nếu landmark nằm trong frame

                    if (
                        0 <= x < width
                        and
                        0 <= y < height
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                # DRAW CONNECTIONS

                for start_idx, end_idx in POSE_CONNECTIONS:

                    start = pose_landmarks[
                        start_idx
                    ]

                    end = pose_landmarks[
                        end_idx
                    ]


                    x1 = int(
                        start.x * width
                    )

                    y1 = int(
                        start.y * height
                    )


                    x2 = int(
                        end.x * width
                    )

                    y2 = int(
                        end.y * height
                    )


                    if (
                        0 <= x1 < width
                        and
                        0 <= y1 < height
                        and
                        0 <= x2 < width
                        and
                        0 <= y2 < height
                    ):

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            2
                        )

        # SHOW FRAME NUMBER

        cv2.putText(
            frame,

            f"Frame: {frame_index}",

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2
        )
        # WRITE OUTPUT VIDEO

        writer.write(
            frame
        )
        # DISPLAY

        cv2.imshow(
            "MediaPipe Pose",
            frame
        )

        # FRAME INDEX
        frame_index += 1

        # PRESS Q TO STOP

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# RELEASE

cap.release()

writer.release()

cv2.destroyAllWindows()


print()

print("================================")
print("DONE")
print("================================")

print(
    f"Processed frames: {frame_index}"
)

print(
    f"Output: {OUTPUT_PATH}"
)