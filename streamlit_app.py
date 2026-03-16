import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model = YOLO("best.pt")

COIN_VALUE = {"1": 1, "2": 2, "5": 5, "10": 10}

st.title("🪙 Coin Detection")
st.write("ตรวจจับและนับเหรียญอัตโนมัติ")

uploaded = st.file_uploader("อัปโหลดรูปเหรียญ", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    img_array = np.array(img)
    results = model(img_array)
    count = {"1": 0, "2": 0, "5": 0, "10": 0}
    for result in results:
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            if name in count:
                count[name] += 1
    total = sum(count[k] * COIN_VALUE[k] for k in count)
    annotated = results[0].plot()
    st.image(annotated, caption="ผลการตรวจจับ")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1 บาท", f"{count['1']} เหรียญ")
    col2.metric("2 บาท", f"{count['2']} เหรียญ")
    col3.metric("5 บาท", f"{count['5']} เหรียญ")
    col4.metric("10 บาท", f"{count['10']} เหรียญ")
    st.success(f"💰 ยอดรวม: {total} บาท")