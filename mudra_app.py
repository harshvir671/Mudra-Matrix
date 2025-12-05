import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import gradio as gr
import numpy as np

# Load your trained model
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['idx_to_class']

# Load model and class mappings
model, idx_to_class = load_model('fast_mudra_classifier.pth')

# Define transformations (same as during training)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Helper function to load image from file path
def load_image(image_input):
    if isinstance(image_input, str):
        # It's a file path
        return Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        # It's a numpy array (from webcam)
        return Image.fromarray(image_input)
    else:
        # It's already a PIL Image
        return image_input

# Prediction function - returns formatted string
def predict_mudra(image_input):
    if image_input is None:
        return "Please upload an image to analyze"
    
    try:
        # Load and convert image
        image = load_image(image_input)
        
        # Preprocess image
        input_tensor = val_transform(image).unsqueeze((0))
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_prob, top_index = torch.max(probabilities, 1)
        
        # Get top prediction with percentage
        predicted_class = idx_to_class[top_index.item()]
        confidence_percent = top_prob.item() * 100
        
        # Format as clean result without JSON formatting
        if confidence_percent > 70:
            confidence_text = "Highly Confident"
        elif confidence_percent > 40:
            confidence_text = "Moderately Confident"
        else:
            confidence_text = "Low Confidence"
            
        result = f"""
        🧘 PREDICTION RESULT
        
        Mudra: {predicted_class}
        Confidence: {confidence_percent:.1f}% ({confidence_text})
        
        """
        
        return result
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Function to display uploaded image
def show_uploaded_image(image_input):
    if image_input is None:
        return None
    
    try:
        image = load_image(image_input)
        image = image.resize((300, 300))
        return image
    except:
        return None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Bharatanatyam Mudra Recognition System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧘 Bharatanatyam Mudra Recognition System")
        gr.Markdown("Upload an image of a hand gesture to identify which mudra it represents.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Image")
                file_input = gr.Image(type="filepath", label="Select an image")
                gr.Examples(
                    examples=[[r"C:\Users\harta\Desktop\Mudra Recognisation\download.jpeg"]],
                    inputs=file_input,
                    label="Example Image"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Analysis Results")
                output_text = gr.Textbox(
                    label="Prediction", 
                    lines=5,
                    placeholder="Results will appear here after image upload..."
                )
                output_image = gr.Image(
                    label="Uploaded Image Preview", 
                    interactive=False,
                    height=300
                )
        
        # Connect the input to the functions
        file_input.change(
            fn=predict_mudra,
            inputs=file_input,
            outputs=output_text
        )
        
        file_input.change(
            fn=show_uploaded_image,
            inputs=file_input,
            outputs=output_image
        )
        
        # Add some info text
        gr.Markdown("---")
        gr.Markdown("""
        ### How to use:
        1. Upload an image of a hand gesture or mudra by clicking the attachment/upload button.
        (or)
        Use your camera to capture a picture and then upload it here the same way.
        
        2. The system will analyze the image
        3. Results will show the predicted mudra with confidence level
        
        ### Tips for best results:
        - Use clear, well-lit images
        - Focus on the hand gesture
        - Plain backgrounds work best
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False
    )
