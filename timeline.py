import streamlit as st


def render_timeline_item(
    container,
    item,
    thumbnail_width=125
):

    with container:

        st.markdown(
            '<div class="timeline-card">',
            unsafe_allow_html=True
        )

        image_col, info_col = st.columns(
            [1, 1.45],
            vertical_alignment="center"
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        with image_col:

            st.image(
                item["image"],
                width=thumbnail_width
            )

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        with info_col:

            st.markdown(
                f'<div class="timeline-time">'
                f'{item["time"]:.2f}s · '
                f'Frame {item["frame"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="timeline-action">'
                f'🎯 {item["action"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f'GMM states: {item["states"]}'
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )