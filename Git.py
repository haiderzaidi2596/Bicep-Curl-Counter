import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Load model once and cache it for performance
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")


movenet_model = load_model()


class BicepCounterTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.movenet = movenet_model.signatures['serving_default']

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # AI Processing
        tf_img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
        outputs = self.movenet(tf.cast(tf_img, dtype=tf.int32))
        keypoints = outputs['output_0'].numpy()[0][0]

        # Left Arm Logic (Shoulder: 5, Elbow: 7, Wrist: 9)
        if keypoints[7][2] > 0.3:
            shoulder = [keypoints[5][1] * w, keypoints[5][0] * h]
            elbow = [keypoints[7][1] * w, keypoints[7][0] * h]
            wrist = [keypoints[9][1] * w, keypoints[9][0] * h]

            angle = self.calculate_angle(shoulder, elbow, wrist)

            # State Machine
            if angle > 150: self.stage = "DOWN"
            if angle < 40 and self.stage == "DOWN":
                self.stage = "UP"
                self.counter += 1

            # Draw on Frame
            cv2.putText(img, f"Reps: {self.counter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Stage: {self.stage}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img


st.title("AI Bicep Curl Counter")
webrtc_streamer(key="bicep-counter", video_transformer_factory=BicepCounterTransformer)