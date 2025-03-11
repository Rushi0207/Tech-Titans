import kagglehub

# Download latest version
path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")

print("Path to dataset files:", path)