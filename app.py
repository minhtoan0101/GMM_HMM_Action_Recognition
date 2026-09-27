import os
import tempfile

import streamlit as st

from config import ACTIONS
from styles import (
    apply_custom_css,
    render_header
)
from models import load_models
from timeline import render_timeline_item
from video_viewer import (
    render_interactive_viewer
)
from video_analyzer import (
    analyze_video
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GMM + HMM Action Recognition",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# UI STYLE
# ============================================================

apply_custom_css()

render_header()


# ============================================================
# MAIN 3-COLUMN LAYOUT
# ============================================================

left_col, center_col, right_col = st.columns(
    [1.05, 2.45, 1.35],
    gap="medium"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

    st.markdown(
        '<div class="section-title">'
        '⚙️ Recognition Settings'
        '</div>',
        unsafe_allow_html=True
    )

    window_size = st.slider(
        "Frame",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help=(
            "Số frame được đưa vào một "
            "sliding window để GMM + HMM "
            "nhận diện."
        )
    )

    step_size = st.slider(
        "Prediction Interval",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help=(
            "Cứ sau bao nhiêu frame thì "
            "hệ thống thực hiện một lần "
            "dự đoán."
        )
    )

    st.markdown("---")

    st.markdown("**Pipeline**")

    st.caption(
        "Video → MediaPipe Pose → 99D → "
        "Sliding Window → GMM → HMM"
    )

    st.markdown("---")

    st.markdown("**Action Classes**")

    for action in ACTIONS:

        st.write(
            f"• {action}"
        )


# ============================================================
# CENTER COLUMN
# ============================================================

with center_col:

    st.markdown(
        '<div class="section-title">'
        '🎬 Video Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload video",
        type=[
            "avi",
            "mp4",
            "mov"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file:

        st.caption(
            "🎞️ Video sẽ chạy tuần tự "
            "trong quá trình phân tích."
        )

        analyze = st.button(
            "▶️ BẮT ĐẦU PHÂN TÍCH",
            use_container_width=True,
            type="primary"
        )

        if analyze:

            # ==================================================
            # SAVE UPLOADED VIDEO
            # ==================================================

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                video_path = (
                    temp_file.name
                )

            try:

                # ==============================================
                # LOAD MODELS
                # ==============================================

                with st.spinner(
                    "Đang tải GMM + HMM..."
                ):

                    (
                        scaler,
                        gmm,
                        hmm_models
                    ) = load_models()

                # ==============================================
                # UI PLACEHOLDERS
                # ==============================================

                progress = st.progress(
                    0,
                    text=(
                        "Đang chuẩn bị "
                        "phân tích..."
                    )
                )

                # ==============================================
                # ACTION TIMELINE
                # ==============================================

                with right_col:

                    st.markdown(
                        '<div class="section-title">'
                        '🕒 Action Timeline'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    timeline_status = (
                        st.empty()
                    )

                    timeline_container = (
                        st.container(
                            height=650,
                            border=True
                        )
                    )

                # ==============================================
                # PROGRESS CALLBACK
                # ==============================================

                def update_progress(
                    progress_value,
                    current_frame,
                    total_frames
                ):

                    progress.progress(
                        progress_value,
                        text=(
                            f"Analyzing frame "
                            f"{current_frame:,}/"
                            f"{total_frames:,}"
                        )
                    )

                # ==============================================
                # PREDICTION CALLBACK
                # ==============================================

                def update_prediction(
                    item,
                    prediction_count
                ):

                    render_timeline_item(
                        timeline_container,
                        item
                    )

                    timeline_status.info(
                        f"{prediction_count} "
                        f"prediction(s) · "
                        f"latest: "
                        f"{item['time']:.2f}s → "
                        f"{item['action']}"
                    )

                # ==============================================
                # RUN ANALYSIS
                # ==============================================

                result = analyze_video(

                    video_path,

                    window_size,

                    step_size,

                    scaler,

                    gmm,

                    hmm_models,

                    progress_callback=(
                        update_progress
                    ),

                    prediction_callback=(
                        update_prediction
                    )
                )

                # ==============================================
                # FINISH PROGRESS
                # ==============================================

                progress.progress(
                    1.0,
                    text=(
                        "Phân tích hoàn tất"
                    )
                )

                # ==============================================
                # VIDEO VIEWER
                # ==============================================

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '🎞️ Video Viewer'
                    '</div>',
                    unsafe_allow_html=True
                )

                render_interactive_viewer(

                    result[
                        "preview_frames"
                    ],

                    result[
                        "fps"
                    ],

                    result[
                        "duration"
                    ],

                    result[
                        "total_frames"
                    ]
                )

                # ==============================================
                # ANALYSIS SUMMARY
                # ==============================================

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '📊 Analysis Summary'
                    '</div>',
                    unsafe_allow_html=True
                )

                # ----------------------------------------------
                # ROW 1
                # ----------------------------------------------

                (
                    summary_1,
                    summary_2,
                    summary_3
                ) = st.columns(3)

                with summary_1:

                    st.metric(
                        "Frames",
                        f"{result['total_frames']:,}"
                    )

                with summary_2:

                    st.metric(
                        "GMM States",
                        f"{result['gmm_state_count']}"
                    )

                with summary_3:

                    st.metric(
                        "Time",
                        f"{result['duration']:.2f}s"
                    )

                # ----------------------------------------------
                # ROW 2
                # ----------------------------------------------

                (
                    summary_4,
                    summary_5,
                    summary_6
                ) = st.columns(3)

                with summary_4:

                    st.metric(
                        "Predictions",
                        f"{len(result['prediction_history']):,}"
                    )

                with summary_5:

                    st.metric(
                        "FPS",
                        f"{result['fps']:.2f}"
                    )

                with summary_6:

                    st.metric(
                        "Processing Time",
                        f"{result['analysis_time']:.2f}s"
                    )

            except Exception as e:

                st.error(
                    f"Lỗi: {e}"
                )

            finally:

                # ==============================================
                # DELETE TEMP VIDEO
                # ==============================================

                if os.path.exists(
                    video_path
                ):

                    os.remove(
                        video_path
                    )

    else:

        st.info(
            "📁 Hãy upload một video "
            "để bắt đầu."
        )

        st.markdown(
            """
            **Các định dạng hỗ trợ:** MP4, AVI, MOV

            Sau khi upload:

            1. Chọn **Frame** và
               **Prediction Interval** ở bên trái.

            2. Nhấn **BẮT ĐẦU PHÂN TÍCH**.

            3. Video được xử lý tuần tự.

            4. Sau khi hoàn tất,
               **Video Viewer** cho phép
               Play/Pause, tua và zoom.

            5. Kết quả prediction xuất hiện
               ở **Action Timeline** bên phải.
            """
        )


# ============================================================
# RIGHT COLUMN - INITIAL STATE
# ============================================================

with right_col:

    if not uploaded_file:

        st.markdown(
            '<div class="section-title">'
            '🕒 Action Timeline'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Timeline sẽ hiển thị thời gian, "
            "action và ảnh snapshot "
            "của từng prediction."
        )