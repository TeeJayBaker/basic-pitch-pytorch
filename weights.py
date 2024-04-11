"""
Load the Tensorflow model weights from basic-pitch and transpose them to PyTorch
"""

import tensorflow as tf
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

audio_path = "01_BN2-131-B_solo_mic.wav"
model_output, midi_data, note_events = predict(audio_path, basic_pitch_model)

print(model_output)
