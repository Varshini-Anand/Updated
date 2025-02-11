from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

# Suppress future warnings globally (optional)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Initialize the Flask app
app = Flask(__name__)

# Set up the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained EfficientNetB0 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(weights=None)  # Use weights=None explicitly
num_classes = 4  # Replace with the number of classes in your dataset
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(r'C:\Users\91984\Desktop\Project\static\models\efficientnet_best_model.pth', map_location=device, weights_only=False))
model = model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_eye_disease(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_class = torch.max(outputs, 1)

        class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']  # Replace with your class names
        return class_names[predicted_class.item()]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error in prediction"

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Make a prediction
            prediction = predict_eye_disease(image_path)

    return render_template('index.html', prediction=prediction, image=os.path.basename(image_path) if image_path else None)

if __name__ == '__main__':
    app.run(debug=True)
