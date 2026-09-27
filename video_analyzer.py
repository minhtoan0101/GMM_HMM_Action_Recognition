import time
import cv2
import numpy as np

from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import POSE_MODEL
from pose_utils import extract_pose_from_frame
from predictor import predict_window


def analyze_video(
    video_path,
    window_size,
    step_size,
    scaler,
    gmm,
    hmm_models,
    progress_callback=None,
    prediction_callback=None
):

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Không thể mở video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    # ========================================================
    # PREVIEW FRAMES
    # ========================================================

    preview_frames = []

    max_preview_frames = 300

    preview_stride = max(
        1,
        int(
            np.ceil(
                total_frames
                /
                max_preview_frames
            )
        )
    )

    # ========================================================
    # POSE LANDMARKER
    # ========================================================

    BaseOptions = python.BaseOptions

    options = (
        vision.PoseLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=POSE_MODEL
            ),

            running_mode=(
                vision.RunningMode.VIDEO
            ),

            num_poses=1
        )
    )

    landmarker = (
        vision.PoseLandmarker
        .create_from_options(
            options
        )
    )

    # ========================================================
    # BUFFERS
    # ========================================================

    feature_buffer = deque(
        maxlen=window_size
    )

    prediction_history = []

    frame_index = 0

    gmm_state_count = getattr(
        gmm,
        "n_components",
        0
    )

    analysis_start = (
        time.perf_counter()
    )

    # ========================================================
    # FRAME LOOP
    # ========================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            timestamp_ms = int(
                frame_index
                * 1000
                / fps
            )

            current_time = (
                frame_index / fps
            )

            # ------------------------------------------------
            # SAVE PREVIEW FRAME
            # ------------------------------------------------

            if (
                frame_index
                % preview_stride
                == 0
            ):

                preview = frame.copy()

                max_width = 720

                h, w = (
                    preview.shape[:2]
                )

                if w > max_width:

                    new_h = int(
                        h
                        * max_width
                        / w
                    )

                    preview = cv2.resize(
                        preview,
                        (
                            max_width,
                            new_h
                        ),
                        interpolation=(
                            cv2.INTER_AREA
                        )
                    )

                ok, encoded = (
                    cv2.imencode(
                        ".jpg",
                        preview,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            72
                        ]
                    )
                )

                if ok:

                    preview_frames.append(
                        encoded.tobytes()
                    )

            # ------------------------------------------------
            # POSE EXTRACTION
            # ------------------------------------------------

            pose_vector = (
                extract_pose_from_frame(
                    frame,
                    landmarker,
                    timestamp_ms
                )
            )

            if pose_vector is not None:

                feature_buffer.append(
                    pose_vector
                )

            # ------------------------------------------------
            # SLIDING WINDOW
            # ------------------------------------------------

            if (

                len(feature_buffer)
                >= window_size

                and

                frame_index
                % step_size
                == 0

            ):

                window_array = np.array(
                    feature_buffer
                )

                (
                    current_prediction,
                    scores,
                    current_states
                ) = predict_window(

                    window_array,

                    scaler,

                    gmm,

                    hmm_models

                )

                # Number of observed states
                observed_states = len(
                    np.unique(
                        current_states
                    )
                )

                # ------------------------------------------------
                # SNAPSHOT
                # ------------------------------------------------

                snapshot = cv2.cvtColor(
                    frame.copy(),
                    cv2.COLOR_BGR2RGB
                )

                prediction_item = {

                    "time":
                        current_time,

                    "frame":
                        frame_index,

                    "action":
                        current_prediction,

                    "states":
                        observed_states,

                    "image":
                        snapshot,

                    "scores":
                        scores
                }

                prediction_history.append(
                    prediction_item
                )

                # ------------------------------------------------
                # CALLBACK
                # ------------------------------------------------

                if prediction_callback:

                    prediction_callback(
                        prediction_item,
                        len(
                            prediction_history
                        )
                    )

            # ------------------------------------------------
            # PROGRESS CALLBACK
            # ------------------------------------------------

            progress_value = (
                (frame_index + 1)
                /
                max(
                    total_frames,
                    1
                )
            )

            if progress_callback:

                progress_callback(
                    min(
                        progress_value,
                        1.0
                    ),
                    frame_index + 1,
                    total_frames
                )

            frame_index += 1

    finally:

        cap.release()

        landmarker.close()

    # ========================================================
    # PROCESSING TIME
    # ========================================================

    analysis_time = (
        time.perf_counter()
        -
        analysis_start
    )

    return {

        "fps":
            fps,

        "total_frames":
            total_frames,

        "duration":
            duration,

        "preview_frames":
            preview_frames,

        "prediction_history":
            prediction_history,

        "gmm_state_count":
            gmm_state_count,

        "analysis_time":
            analysis_time
    }