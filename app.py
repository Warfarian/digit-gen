import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Define the model architecture (same as training)
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder - matches training script exactly
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder - matches training script exactly
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def decode(self, z, labels):
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        z_labeled = torch.cat([z, labels_onehot], dim=1)
        return self.decoder(z_labeled)

@st.cache_resource
def load_model():
    """Load the trained Conditional VAE model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with same architecture as training
    model = ConditionalVAE(
        input_dim=784, 
        hidden_dim=512, 
        latent_dim=20, 
        num_classes=10
    ).to(device)
    
    try:
        # Load the trained model weights
        checkpoint = torch.load('models/cvae_mnist_final.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        st.success("✅ Conditional VAE model loaded successfully!")
        return model, device
    except FileNotFoundError:
        st.error("❌ Model file 'cvae_mnist_final.pth' not found. Please upload the trained model file.")
        st.info("💡 Run the training script first to generate the model file.")
        return None, device

def generate_digits(model, device, digit_class, num_samples=5):
    """
    Generate specific digit images using the trained Conditional VAE
    
    This is the key function - it generates SPECIFIC digits by conditioning
    the VAE on the digit label (0-9)
    """
    if model is None:
        return None
    
    model.eval()
    with torch.no_grad():
        # Create labels for the specific digit we want to generate
        # This is how we control what digit gets generated!
        labels = torch.full((num_samples,), digit_class, dtype=torch.long).to(device)
        
        # Sample random noise from latent space
        z = torch.randn(num_samples, model.latent_dim).to(device)
        
        # Generate images conditioned on the digit class
        # The model uses the label to determine what digit to generate
        generated = model.decode(z, labels)
        
        # Reshape to image format and ensure proper range
        generated = generated.view(-1, 28, 28).cpu().numpy()
        generated = np.clip(generated, 0, 1)
        
        return generated

def create_image_grid(images, digit_class):
    """Create a grid of generated images"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f'Generated Digit: {digit_class}', fontsize=16, fontweight='bold')
    
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=12)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def main():
    # Title and description
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("""
    This web application generates **specific** handwritten digit images using a **Conditional Variational Autoencoder (CVAE)** 
    trained on the MNIST dataset. 
    
    **How it works:** Select any digit (0-9) and the AI model will generate 5 unique handwritten samples of that exact digit!
    
    🎯 **Key Feature**: This isn't random generation - you control exactly which digit gets generated.
    """)
    
    # Add info box explaining the conditional generation
    st.info("""
    🧠 **About Conditional Generation**: Unlike regular VAEs that generate random samples, this Conditional VAE 
    takes the digit label (0-9) as input, allowing it to generate specific digits on demand.
    """)
    
    st.info("👉 Use the sidebar to select a digit and generate new samples!", icon="⚙️")
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for controls
    st.sidebar.header("Generation Controls")
    
    # Digit selection
    digit_class = st.sidebar.selectbox(
        "Select Digit to Generate:",
        options=list(range(10)),
        index=0,
        help="Choose which digit (0-9) you want to generate"
    )
    
    # Generation button
    if st.sidebar.button("🎲 Generate New Samples", type="primary"):
        with st.spinner(f"Generating digit {digit_class}..."):
            # Generate images
            generated_images = generate_digits(model, device, digit_class, num_samples=5)
            
            if generated_images is not None:
                # Store in session state
                st.session_state.generated_images = generated_images
                st.session_state.current_digit = digit_class
                st.success(f"Generated 5 samples of digit {digit_class}!")
    
    # Display generated images
    if hasattr(st.session_state, 'generated_images') and st.session_state.generated_images is not None:
        st.subheader(f"Generated Samples for Digit: {st.session_state.current_digit}")
        
        # Create and display image grid
        image_grid = create_image_grid(st.session_state.generated_images, st.session_state.current_digit)
        st.image(image_grid, use_column_width=True)
        
        # Display individual images in columns
        st.subheader("Individual Samples")
        cols = st.columns(5)
        
        for i in range(5):
            with cols[i]:
                # Convert numpy array to PIL Image
                img_array = (st.session_state.generated_images[i] * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                st.image(img, caption=f"Sample {i+1}", use_column_width=True)
    
    else:
        # Display placeholder
        st.info("👆 Select a digit and click 'Generate New Samples' to start generating!")
        
        # Show example of what the app can do
        st.subheader("About This App")
        st.markdown("""
        ### Features:
        - **Conditional Generation**: Generate specific digits (0-9) on demand
        - **Multiple Samples**: Get 5 unique variations of the same digit
        - **MNIST-style Output**: 28x28 grayscale images similar to the original dataset
        - **Deep Learning Model**: Uses a Conditional Variational Autoencoder trained from scratch
        
        ### How it Works:
        1. Select a digit (0-9) from the sidebar
        2. Click "Generate New Samples" 
        3. The AI model creates 5 unique handwritten versions of your chosen digit
        4. View the results in both grid and individual formats
        
        ### Model Details:
        - **Architecture**: Conditional Variational Autoencoder (CVAE)
        - **Training Data**: MNIST dataset (70,000 handwritten digits)
        - **Framework**: PyTorch
        - **Training Environment**: Google Colab with T4 GPU
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Note**: Each generation produces unique samples due to the stochastic nature of the VAE model.")

if __name__ == "__main__":
    main()