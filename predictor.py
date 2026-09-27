import numpy as np

from pose_utils import normalize_pose


def predict_window(
    feature_window,
    scaler,
    gmm,
    hmm_models
):

    # ========================================================
    # 1. NORMALIZATION
    # ========================================================

    normalized = normalize_pose(
        feature_window
    )

    # ========================================================
    # 2. STANDARD SCALER
    # ========================================================

    scaled = scaler.transform(
        normalized
    )

    # ========================================================
    # 3. GMM
    # ========================================================

    states = gmm.predict(
        scaled
    )

    # ========================================================
    # 4. HMM INPUT
    # ========================================================

    observations = states.reshape(
        -1,
        1
    )

    # ========================================================
    # 5. HMM SCORING
    # ========================================================

    scores = {}

    for action, hmm in hmm_models.items():

        try:

            score = hmm.score(
                observations
            )

        except Exception:

            score = -np.inf

        scores[action] = score

    # ========================================================
    # 6. FINAL PREDICTION
    # ========================================================

    prediction = max(
        scores,
        key=scores.get
    )

    return (
        prediction,
        scores,
        states
    )