from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np
import librosa
import pickle  # Needed for loading label encoder

# Importing Local files
from tracker import video_analysis

# Initialize Flask app
app = Flask(__name__)

# Defining model paths
model_path = 'models/yolov8n.pt'  # Removed leading slash for relative path
audio_model_path = 'models/yolov8n.pt'  # You need to specify your audio model path
audio_model = None
label_encoder = None

try:
    # Load your audio model (not YOLO model as shown in original code)
    audio_model = load_model(audio_model_path)
    print("Audio model loaded successfully.")
    
    # Load label encoder (you'll need to save this during training)
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print("Error loading models:", e)

# Audio file path - consider making this configurable
audio_file_path = 'audio_files/Emergency_audio.wav'  # Fixed space in filename

def features_extractor(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def test_audio_lstm(audio_file_path, model):
    if model is None or label_encoder is None:
        return "Model not loaded"
    
    # Extract features from the audio file
    features = features_extractor(audio_file_path)
    if features is None:
        return "Feature extraction failed"
    
    # Reshape the features to match the input shape of the model
    features = features.reshape(1, -1, 80)
    
    try:
        # Make prediction using the model
        prediction = model.predict(features)
        # Decode the predicted class
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_class[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction failed"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/audio')
def audio_simulation():
    return render_template('audio.html')

@app.route('/video')
def video_simulation():
    return render_template('video.html')

@app.route('/ambulance', methods=['GET', 'POST'])
def ambulance_generated():
    emergency_vehicle_detected = None
    time_to_be_added = 0  # Initialize with default value
    
    # Audio detection
    if audio_model is not None:
        emergency_vehicle_detected = test_audio_lstm(audio_file_path=audio_file_path, model=audio_model)
    else:
        print("Audio model not loaded!")
        emergency_vehicle_detected = "Model not loaded"
    
    print(emergency_vehicle_detected)
    
    # Video analysis - uncomment when ready
    try:
        speed, distance = video_analysis()
        if speed < 30 and distance < 20:
            time_to_be_added = 10
        elif speed < 50 and distance < 50:
            time_to_be_added = 20
        else:
            time_to_be_added = 30
    except Exception as e:
        print(f"Video analysis error: {e}")
        time_to_be_added = 30  # Default value if analysis fails
    
    return jsonify({
        'emergency_vehicle_detected': emergency_vehicle_detected,
        'time_to_be_added': time_to_be_added
    })

if __name__ == '__main__':
    app.run(debug=True)