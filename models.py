import os
import joblib
import streamlit as st

from config import MODEL_DIR, ACTIONS


@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # GMM scaler
    # --------------------------------------------------------

    scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "gmm_scaler.pkl"
        )
    )

    # --------------------------------------------------------
    # GMM
    # --------------------------------------------------------

    gmm = joblib.load(
        os.path.join(
            MODEL_DIR,
            "gmm_model.pkl"
        )
    )

    # --------------------------------------------------------
    # HMM models
    # --------------------------------------------------------

    hmm_models = {}

    for action in ACTIONS:

        path = os.path.join(
            MODEL_DIR,
            f"hmm_{action}.pkl"
        )

        hmm_models[action] = joblib.load(
            path
        )

    return scaler, gmm, hmm_models