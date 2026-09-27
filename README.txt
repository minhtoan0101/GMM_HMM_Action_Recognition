Ứng dụng GMM kết hợp HMM trong nhận diện hành động người
============================================================
Mô tả:
Hệ thống nhận diện hành động người từ video bằng cách
trích xuất tư thế người (pose) với MediaPipe Pose, biểu diễn
mỗi frame thành vector 99 đặc trưng, sử dụng GMM để chuyển
các đặc trưng thành chuỗi trạng thái và sử dụng HMM để mô
hình hóa sự chuyển đổi trạng thái theo thời gian.

------------------------------------------------------------
1. MỤC TIÊU
------------------------------------------------------------

Xây dựng hệ thống nhận diện 5 loại hành động người:

1. BodyWeightSquats
2. JumpingJack
3. PullUps
4. Punch
5. PushUps

Pipeline chính:

Video
  ↓
MediaPipe Pose
  ↓
33 Body Landmarks
  ↓
99 Features / Frame
  ↓
Normalization
  ↓
StandardScaler
  ↓
GMM
  ↓
State Sequence
  ↓
5 HMM
  ↓
Log-Likelihood
  ↓
Predicted Action


------------------------------------------------------------
2. Ý TƯỞNG GMM + HMM
------------------------------------------------------------

GMM (Gaussian Mixture Model):
- Mô hình hóa sự phân bố của các đặc trưng pose.
- Mỗi frame được ánh xạ vào một Gaussian component.
- Gaussian component được sử dụng như một trạng thái trung
  gian của tư thế.

HMM (Hidden Markov Model):
- Nhận chuỗi trạng thái được tạo ra từ GMM.
- Mô hình hóa sự chuyển đổi của các trạng thái theo thời gian.
- Mỗi hành động có một HMM riêng.
- HMM tính log-likelihood của chuỗi trạng thái.
- Hành động có log-likelihood cao nhất được chọn làm kết quả.

Tóm tắt:

GMM xử lý đặc trưng ở mức frame/tư thế.
HMM xử lý mối quan hệ theo thời gian.


------------------------------------------------------------
3. CÁC HÀNH ĐỘNG
------------------------------------------------------------

Hệ thống hiện tại gồm 5 action classes:

- BodyWeightSquats
- JumpingJack
- PullUps
- Punch
- PushUps


------------------------------------------------------------
4. CẤU TRÚC PROJECT
------------------------------------------------------------

GMM_HMM_Action_Recognition/
│
├── app.py
│   └── File chạy chính của ứng dụng Streamlit.
│
├── config.py
│   └── Cấu hình project, đường dẫn model và action classes.
│
├── styles.py
│   └── CSS và các thành phần giao diện.
│
├── models.py
│   └── Load GMM, Scaler và các HMM.
│
├── pose_utils.py
│   └── Xử lý MediaPipe Pose,
│       trích xuất 33 landmarks,
│       tạo vector 99 chiều,
│       chuẩn hóa pose.
│
├── predictor.py
│   └── Logic dự đoán:
│       Normalize → Scaler → GMM → State Sequence
│       → HMM → Log-Likelihood → Action.
│
├── video_analyzer.py
│   └── Đọc và xử lý video,
│       trích xuất pose theo từng frame,
│       sliding window và gọi predictor.
│
├── timeline.py
│   └── Hiển thị Action Timeline.
│
├── video_viewer.py
│   └── Hiển thị Video Viewer,
│       Play/Pause, tua video và Zoom.
│
├── models/
│   └── pose_landmarker.task
│       MediaPipe Pose Landmarker model.
│
└── output/
    └── models/
        ├── gmm_scaler.pkl
        ├── gmm_model.pkl
        ├── hmm_BodyWeightSquats.pkl
        ├── hmm_JumpingJack.pkl
        ├── hmm_PullUps.pkl
        ├── hmm_Punch.pkl
        └── hmm_PushUps.pkl


------------------------------------------------------------
5. ĐẶC TRƯNG DỮ LIỆU
------------------------------------------------------------

MediaPipe Pose cung cấp 33 landmarks.

Mỗi landmark gồm:

x, y, z

Do đó:

33 landmarks × 3 tọa độ = 99 features / frame

Một video gồm nhiều frame nên được biểu diễn dưới dạng
chuỗi thời gian:

X = [X1, X2, ..., XT]

Trong đó:

Xt ∈ R^99

Toàn bộ video có dạng:

X ∈ R^(T × 99)


------------------------------------------------------------
6. NORMALIZATION
------------------------------------------------------------

Để giảm ảnh hưởng của vị trí và kích thước cơ thể,
pose được chuẩn hóa theo hai bước:

Bước 1:
Lấy trung điểm của hai hông làm tâm.

Hip Center =
(Left Hip + Right Hip) / 2

Sau đó dịch toàn bộ landmark theo tâm này.

Bước 2:
Scale theo khoảng cách giữa hai vai.

Shoulder Distance =
distance(Left Shoulder, Right Shoulder)

Các tọa độ sau đó được chia cho khoảng cách này.


------------------------------------------------------------
7. GMM
------------------------------------------------------------

GMM được sử dụng để chuyển vector pose của từng frame
thành một trạng thái.

Quy trình:

99D Feature
    ↓
StandardScaler
    ↓
GMM
    ↓
GMM State

Các Gaussian components của GMM không tương ứng trực tiếp
với số lượng action classes.

Chúng đóng vai trò là các trạng thái trung gian biểu diễn
các cấu hình/tư thế khác nhau của cơ thể.


------------------------------------------------------------
8. HMM
------------------------------------------------------------

Mỗi action class có một HMM riêng.

Ví dụ:

HMM BodyWeightSquats
HMM JumpingJack
HMM PullUps
HMM Punch
HMM PushUps

Chuỗi trạng thái từ GMM được đưa vào từng HMM.

Ví dụ:

GMM:
Frame 1 → State 0
Frame 2 → State 0
Frame 3 → State 2
Frame 4 → State 3
Frame 5 → State 3
...

Tạo thành:

[0, 0, 2, 3, 3, ...]

Chuỗi này được đưa vào 5 HMM.

Mỗi HMM trả về một log-likelihood.

Hệ thống chọn action có log-likelihood cao nhất.


------------------------------------------------------------
9. SLIDING WINDOW
------------------------------------------------------------

Hệ thống không nhất thiết dự đoán toàn bộ video chỉ bằng
một lần.

Video được chia thành các cửa sổ thời gian (sliding window).

Ví dụ:

Window 1:
Frame 1 → Frame N

Window 2:
Frame K → Frame K+N

Window 3:
Frame 2K → Frame 2K+N

Mỗi window được đưa qua:

Pose
→ Normalize
→ GMM
→ State Sequence
→ HMM
→ Prediction

Các prediction được sử dụng để tạo Action Timeline.

------------------------------------------------------------
12. YÊU CẦU MÔI TRƯỜNG
------------------------------------------------------------

Python:
Python 3.x

Các thư viện chính:

- Streamlit
- OpenCV
- NumPy
- MediaPipe
- scikit-learn
- hmmlearn
- joblib

Nếu project có file requirements.txt thì cài đặt bằng:

pip install -r requirements.txt

------------------------------------------------------------
13. CÀI ĐẶT
------------------------------------------------------------

Bước 1:
Mở Terminal / PowerShell.

Bước 2:
Di chuyển đến thư mục project.

Ví dụ:

cd "D:\Duong_Dan\GMM_HMM_Action_Recognition"

Bước 3:
Tạo virtual environment:

python -m venv .venv

Bước 4:
Kích hoạt virtual environment trên Windows:

.venv\Scripts\activate
.venv\Scripts\Activate.ps1

Bước 5:
Cài đặt thư viện:

pip install -r requirements.txt


------------------------------------------------------------
14. CHẠY ỨNG DỤNG
------------------------------------------------------------

Sau khi cài đặt xong:

streamlit run app.py

Streamlit sẽ mở giao diện ứng dụng trên trình duyệt.


------------------------------------------------------------
15. SỬ DỤNG ỨNG DỤNG
------------------------------------------------------------

Bước 1:
Upload video.

Bước 2:
Chọn:

- Frame / Window Size
- Prediction Interval

Bước 3:
Nhấn:

BẮT ĐẦU PHÂN TÍCH

Bước 4:
Hệ thống thực hiện:

Video
→ MediaPipe Pose
→ 99 Features
→ Normalize
→ StandardScaler
→ GMM
→ State Sequence
→ HMM
→ Action

Bước 5:
Kết quả được hiển thị tại:

- Action Timeline
- Video Viewer
- Analysis Summary


------------------------------------------------------------
16. VIDEO VIEWER
------------------------------------------------------------

Video Viewer hỗ trợ:

- Play
- Pause
- Previous Frame
- Next Frame
- Rewind 5 seconds
- Forward 5 seconds
- Zoom In
- Zoom Out
- Reset Zoom
- Thanh timeline để di chuyển đến frame


------------------------------------------------------------
17. ACTION TIMELINE
------------------------------------------------------------

Action Timeline hiển thị:

- Thời gian
- Frame
- Action được dự đoán
- Snapshot của frame
- Thông tin trạng thái GMM

Timeline giúp quan sát sự thay đổi của action prediction
theo thời gian.


------------------------------------------------------------
18. LUỒNG HOẠT ĐỘNG TỔNG QUÁT
------------------------------------------------------------

                    VIDEO
                      │
                      ▼
              MediaPipe Pose
                      │
                      ▼
              33 Landmarks
                      │
                      ▼
                99 Features
                      │
                      ▼
                Normalize
                      │
                      ▼
               StandardScaler
                      │
                      ▼
                     GMM
                      │
                      ▼
              State Sequence
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       HMM 1        HMM 2       ... HMM 5
          │           │             │
          └───────────┼─────────────┘
                      ▼
              Log-Likelihood
                      │
                      ▼
              Highest Score
                      │
                      ▼
              ACTION PREDICTION


------------------------------------------------------------
19. CÁC FILE QUAN TRỌNG
------------------------------------------------------------

Muốn sửa giao diện:
→ app.py
→ styles.py

Muốn sửa MediaPipe / pose:
→ pose_utils.py

Muốn sửa thuật toán GMM + HMM:
→ predictor.py

Muốn sửa quá trình đọc và phân tích video:
→ video_analyzer.py

Muốn sửa Action Timeline:
→ timeline.py

Muốn sửa Video Viewer:
→ video_viewer.py

Muốn sửa đường dẫn/model/action:
→ config.py

Muốn sửa cách load model:
→ models.py


------------------------------------------------------------
20. LƯU Ý
------------------------------------------------------------

1. Không đổi tên các file model nếu code đang sử dụng
   tên file hiện tại.

2. Đảm bảo file MediaPipe:

   models/pose_landmarker.task

   tồn tại trước khi chạy ứng dụng.

3. Đảm bảo các file GMM/HMM tồn tại trong:

   output/models/

4. Chạy ứng dụng bằng:

   streamlit run app.py

5. Không chạy trực tiếp predictor.py hoặc
   video_analyzer.py bằng Streamlit.


============================================================
END OF README
============================================================