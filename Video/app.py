import argparse
import os
import re
import time

import torch
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
from training.zoo.classifiers import DeepFakeClassifier

import gradio as gr

def model_fn(model_dir):
	model_path = os.path.join(model_dir, 'b7_ns_best.pth')
	model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns") # default: CPU
	checkpoint = torch.load(model_path, map_location="cpu")
	state_dict = checkpoint.get("state_dict", checkpoint)
	model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
	model.eval()
	del checkpoint
	#models.append(model.half())

	return model

def convert_result(pred, class_names=["Real", "Fake"]):
	preds = [pred, 1 - pred]
	assert len(class_names) == len(preds), "Class / Prediction should have the same length"
	return {n: float(p) for n, p in zip(class_names, preds)}

def predict_fn(video):
	start = time.time()
	prediction = predict_on_video(face_extractor=meta["face_extractor"],
							   video_path=video,
							   batch_size=meta["fps"],
							   input_size=meta["input_size"],
							   models=model,
							   strategy=meta["strategy"],
							   apply_compression=False,
							   device='cpu')

	elapsed_time = round(time.time() - start, 2)

	prediction = convert_result(prediction)

	return prediction, elapsed_time

# Create title, description and article strings
title = "Deepfake Video Detector"

# 
model_dir = 'weights'
frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)
input_size = 380
strategy = confident_strategy
class_names = ["Real", "Fake"]

meta = {"fps": 32,
		"face_extractor": face_extractor,
		"input_size": input_size,
		"strategy": strategy}

model = model_fn(model_dir)

"""
if __name__ == '__main__':
	video_path = "examples/nlurbvsozt.mp4"
	model = model_fn(model_dir)
	a, b = predict_fn(video_path)
	print(a, b)
"""
# Create the Gradio demo
demo = gr.Interface(fn=predict_fn, # mapping function from input to output
					inputs=gr.Video(),
					outputs=[gr.Label(num_top_classes=2, label="Predictions"), # what are the outputs?
							 gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
					title=title)

# Launch the demo!
demo.launch(debug=False,) # Hugging face space don't need shareable_links
