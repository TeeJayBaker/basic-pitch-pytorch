"""
Load the Tensorflow model weights from basic-pitch and transpose them to PyTorch
"""

import tensorflow as tf
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

from model import basic_pitch_torch
import torch

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

# Transpose the weights from the Tensorflow model to the PyTorch model
conv_key_matches = {
    "conv2d_1": "contour_1.0",
    "contours-reduced": "contour_2.0",
    "conv2d_2": "note_1.0",
    "conv2d_3": "note_1.2",
    "conv2d_4": "onset_1.0",
    "conv2d_5": "onset_2.0",
}

batch_key_matches = {
    "batch_normalization_2": "contour_1.1",
    "batch_normalization_3": "onset_1.1",
}

new_state_dict = {}
for key in conv_key_matches.keys():
    new_state_dict[conv_key_matches[key] + ".weight"] = torch.tensor(
        tf_weights_dict[key + "/kernel:0"].transpose(3, 2, 1, 0)
    )
    new_state_dict[conv_key_matches[key] + ".bias"] = torch.tensor(
        tf_weights_dict[key + "/bias:0"]
    )

for key in batch_key_matches.keys():
    new_state_dict[batch_key_matches[key] + ".weight"] = torch.tensor(
        tf_weights_dict[key + "/gamma:0"]
    )
    new_state_dict[batch_key_matches[key] + ".bias"] = torch.tensor(
        tf_weights_dict[key + "/beta:0"]
    )
    new_state_dict[batch_key_matches[key] + ".running_mean"] = torch.tensor(
        tf_weights_dict[key + "/moving_mean:0"]
    )
    new_state_dict[batch_key_matches[key] + ".running_var"] = torch.tensor(
        tf_weights_dict[key + "/moving_variance:0"]
    )

new_state_dict["cqt.lowpass_filter"] = old_state_dict["cqt.lowpass_filter"]
new_state_dict["cqt.lenghts"] = old_state_dict["cqt.lenghts"]
new_state_dict["cqt.cqt_kernels_real"] = old_state_dict["cqt.cqt_kernels_real"]
new_state_dict["cqt.cqt_kernels_imag"] = old_state_dict["cqt.cqt_kernels_imag"]

torch_model.load_state_dict(new_state_dict)
