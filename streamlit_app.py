import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import deque
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import threading

lock = threading.Lock()

def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe("focus/status")

def on_message(client, userdata, msg):
    with lock:
        st.session_state.focus_history.append(float(msg.payload.decode()))

if 'focus_history' not in st.session_state:
    st.session_state.focus_history = deque(maxlen=200)

if 'mqtt_client' not in st.session_state:
    mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect("broker.emqx.io", 1883, 60)
    mqtt_client.loop_start()
    st.session_state.mqtt_client = mqtt_client

st.title("Smart Study Focus Tracker Dashboard")
st.markdown("Real-time focus from webcam ML + IoT sim (CSIT136).")

placeholder = st.empty()

with placeholder.container():
    with lock:
        history_snapshot = list(st.session_state.focus_history)

    if history_snapshot:
        df = pd.DataFrame(history_snapshot, columns=['Focus Level'])
        st.line_chart(df)
        avg = np.mean(history_snapshot)
        st.metric("Average Focus", f"{avg:.2f}", delta="Good" if avg > 0.5 else "Take Break")
        focused_pct = sum(1 for v in history_snapshot if v == 1) / len(history_snapshot) * 100
        st.progress(int(focused_pct), text=f"Focused {focused_pct:.0f}% of session")
    else:
        st.info("Run focus_detector.py first to publish data.")

time.sleep(2)
st.rerun()
