"""
Load the Tensorflow model weights from basic-pitch and transpose them to PyTorch
"""

import tensorflow as tf
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

from model import basic_pitch_torch

basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

audio_path = "01_BN2-131-B_solo_mic.wav"
# model_output, midi_data, note_events = predict(audio_path, basic_pitch_model)

# print(model_output)

# Load model weights
tf_model = tf.saved_model.load(ICASSP_2022_MODEL_PATH)

# Save each weight matrix in a dictionary with the key as the layer name
tf_weights_dict = {}
for variable in tf_model.variables:
    tf_weights_dict[variable.name] = variable.numpy()

# print key and shape for tf_weights_dict
print()
print("-----------tf_state_dict:-----------")
for key in tf_weights_dict.keys():
    print(key, tf_weights_dict[key].shape)

# import torch model and load state dict
torch_model = basic_pitch_torch()
old_state_dict = torch_model.state_dict()

# print key and shape for old_state_dict
print()
print("-----------old_state_dict:-----------")
for key in old_state_dict.keys():
    print(key, old_state_dict[key].shape)
