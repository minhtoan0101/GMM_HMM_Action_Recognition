import base64
import json

import streamlit as st
import streamlit.components.v1 as components


def render_interactive_viewer(
    frame_bytes,
    fps,
    duration,
    total_frames
):

    if not frame_bytes:

        st.info(
            "Không có frame để tạo Video Viewer."
        )

        return

    # ========================================================
    # CONVERT FRAMES TO BASE64
    # ========================================================

    images = [

        "data:image/jpeg;base64,"
        + base64.b64encode(frame).decode("ascii")

        for frame in frame_bytes
    ]

    payload = json.dumps(
        images
    )

    # ========================================================
    # HTML + JAVASCRIPT
    # ========================================================

    html = f"""
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: transparent;
    font-family: Arial, sans-serif;
}}

.viewer {{
    width: 100%;
    background: #111;
    border: 1px solid #333;
    border-radius: 12px;
    overflow: hidden;
}}

.screen {{
    height: 430px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #050505;
    overflow: hidden;
}}

.screen img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    transform-origin: center center;
    transition: transform .12s ease;
    user-select: none;
}}

.controls {{
    padding: 12px 14px 14px;
    background: #181818;
}}

.buttons {{
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    margin-bottom: 9px;
}}

button {{
    border: 1px solid #444;
    background: #292929;
    color: #fff;
    border-radius: 7px;
    padding: 7px 10px;
    cursor: pointer;
    font-size: 13px;
}}

button:hover {{
    background: #3a3a3a;
}}

.time {{
    color: #ddd;
    font-size: 13px;
    margin-bottom: 5px;
    text-align: center;
}}

input[type=range] {{
    width: 100%;
    cursor: pointer;
}}

.zoom {{
    color: #aaa;
    font-size: 12px;
    text-align: right;
    margin-top: 5px;
}}

</style>

</head>


<body>

<div class="viewer">

    <div class="screen">

        <img
            id="frame"
            alt="Sequential video frame"
        >

    </div>


    <div class="controls">

        <div class="buttons">

            <button id="prev">
                ⏮
            </button>

            <button id="rewind">
                ↶ 5s
            </button>

            <button id="play">
                ▶ Play
            </button>

            <button id="forward">
                5s ↷
            </button>

            <button id="next">
                ⏭
            </button>

            <button id="zoomOut">
                −
            </button>

            <button id="zoomReset">
                100%
            </button>

            <button id="zoomIn">
                +
            </button>

        </div>


        <div class="time">

            <span id="time">
                0.00s / {duration:.2f}s
            </span>

        </div>


        <input
            id="timeline"
            type="range"
            min="0"
            max="{max(len(images) - 1, 0)}"
            value="0"
            step="1"
        >


        <div class="zoom">

            Zoom:
            <span id="zoomText">
                100%
            </span>

            · {len(images)} preview frames

        </div>

    </div>

</div>


<script>

const frames = {payload};

const fps = {float(fps)};

const duration = {float(duration)};

const totalFrames = {int(total_frames)};


const img =
    document.getElementById("frame");

const slider =
    document.getElementById("timeline");

const playBtn =
    document.getElementById("play");

const timeText =
    document.getElementById("time");

const zoomText =
    document.getElementById("zoomText");


let index = 0;

let playing = false;

let timer = null;

let zoom = 1;


function currentTime() {{

    return duration > 0

        ? index /
          Math.max(frames.length - 1, 1)
          * duration

        : 0;
}}


function update() {{

    if (!frames.length)
        return;


    index = Math.max(
        0,
        Math.min(
            index,
            frames.length - 1
        )
    );


    img.src = frames[index];

    slider.value = index;


    timeText.textContent =
        currentTime().toFixed(2)
        + "s / "
        + duration.toFixed(2)
        + "s";


    img.style.transform =
        "scale("
        + zoom.toFixed(2)
        + ")";


    zoomText.textContent =
        Math.round(zoom * 100)
        + "%";
}}


function stop() {{

    playing = false;


    if (timer) {{

        clearInterval(timer);

        timer = null;

    }}


    playBtn.textContent =
        "▶ Play";
}}


function start() {{

    if (frames.length <= 1)
        return;


    if (
        index >= frames.length - 1
    )
        index = 0;


    playing = true;

    playBtn.textContent =
        "⏸ Pause";


    const interval = Math.max(

        25,

        (duration * 1000)
        /
        Math.max(
            frames.length - 1,
            1
        )

    );


    timer = setInterval(() => {{

        index += 1;


        if (
            index >= frames.length
        )
            index = 0;


        update();

    }}, interval);
}}


playBtn.addEventListener(
    "click",
    () => {{

        if (playing)
            stop();

        else
            start();

    }}
);


slider.addEventListener(
    "input",
    () => {{

        stop();

        index =
            Number(slider.value);

        update();

    }}
);


document
    .getElementById("prev")
    .addEventListener(
        "click",
        () => {{

            stop();

            index -= 1;


            if (index < 0)
                index =
                    frames.length - 1;


            update();

        }}
    );


document
    .getElementById("next")
    .addEventListener(
        "click",
        () => {{

            stop();

            index += 1;


            if (
                index >= frames.length
            )
                index = 0;


            update();

        }}
    );


function jumpSeconds(
    seconds
) {{

    const ratio =
        duration > 0
            ? seconds / duration
            : 0;


    index += Math.round(

        ratio
        *
        Math.max(
            frames.length - 1,
            1
        )

    );


    index = Math.max(

        0,

        Math.min(
            index,
            frames.length - 1
        )

    );


    update();
}}


document
    .getElementById("rewind")
    .addEventListener(
        "click",
        () => {{

            stop();

            jumpSeconds(-5);

        }}
    );


document
    .getElementById("forward")
    .addEventListener(
        "click",
        () => {{

            stop();

            jumpSeconds(5);

        }}
    );


document
    .getElementById("zoomIn")
    .addEventListener(
        "click",
        () => {{

            zoom =
                Math.min(
                    2.5,
                    zoom + 0.1
                );

            update();

        }}
    );


document
    .getElementById("zoomOut")
    .addEventListener(
        "click",
        () => {{

            zoom =
                Math.max(
                    0.5,
                    zoom - 0.1
                );

            update();

        }}
    );


document
    .getElementById("zoomReset")
    .addEventListener(
        "click",
        () => {{

            zoom = 1;

            update();

        }}
    );


update();

</script>

</body>

</html>
"""

    components.html(
        html,
        height=590,
        scrolling=False
    )