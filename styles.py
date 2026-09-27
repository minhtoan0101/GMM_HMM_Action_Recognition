import streamlit as st


def apply_custom_css():

    st.markdown(
        """
        <style>

        .stAppHeader {
            display: none;
        }

        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }

        .main-title {
            font-size: 2.8rem;
            font-weight: 750;
            margin-top: 0rem;
            margin-bottom: 0.15rem;
            text-align: center;
        }

        .subtitle {
            color: #777;
            margin-bottom: 1rem;
            text-align: center;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.25rem 0 0.8rem 0;
        }

        .timeline-card {
            border: 1px solid #dddddd;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 8px;
            background: rgba(128, 128, 128, 0.06);
        }

        .timeline-time {
            font-size: 0.88rem;
            color: #777;
            margin-bottom: 2px;
        }

        .timeline-action {
            font-size: 1rem;
            font-weight: 700;
        }

        .summary-box {
            border: 1px solid #dddddd;
            border-radius: 10px;
            padding: 12px;
            margin-top: 10px;
        }

        .small-note {
            font-size: 0.82rem;
            color: #777;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():

    st.markdown(
        '<div class="main-title">'
        'GMM + HMM Human Action Recognition'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'MediaPipe Pose → 99D Time Series → '
        'Sliding Window → GMM → HMM'
        '</div>',
        unsafe_allow_html=True
    )