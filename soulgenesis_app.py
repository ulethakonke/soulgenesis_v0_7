# soulgenesis_128_real.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import zipfile
import io

# Set page config
st.set_page_config(
    page_title="SoulGenesis 128 - Real Image Compression",
    page_icon="📸",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6c5ce7;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #a29bfe;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #2d3436;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #00b894;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced CNN Model for 128x128
class EnhancedEncoder(nn.Module):
    def __init__(self, soul_dim=512, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # 128x128 → 64x64
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 → 32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 → 16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 → 8x8
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 → 4x4
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, soul_dim * 2),  # For VAE: mean and variance
        )
    
    def forward(self, x):
        return self.net(x)

class EnhancedDecoder(nn.Module):
    def __init__(self, soul_dim=512, out_channels=3):
        super().__init__()
        self.soul_dim = soul_dim
        self.initial_size = 4
        self.initial_channels = 512
        
        self.fc = nn.Linear(soul_dim, self.initial_channels * self.initial_size * self.initial_size)
        
        self.net = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 8x8 → 16x16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16x16 → 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 → 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 → 128x128
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        return self.net(x)

class VAESoulGenesis(nn.Module):
    def __init__(self, soul_dim=512):
        super().__init__()
        self.soul_dim = soul_dim
        self.encoder = EnhancedEncoder(soul_dim)
        self.decoder = EnhancedDecoder(soul_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode to distribution parameters
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :self.soul_dim], encoded[:, self.soul_dim:]
        
        # Sample from distribution
        soul_seed = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(soul_seed)
        return reconstructed, soul_seed, mu, logvar

# Real dataset loader with caching
@st.cache_resource
def load_real_dataset(dataset_name="cifar10", img_size=128, num_samples=1000):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif dataset_name == "stl10":
        dataset = datasets.STL10(root="./data", split='train', download=True, transform=transform)
    else:  # Use CIFAR-10 as default
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # Use a subset for faster training
    indices = torch.randperm(len(dataset))[:num_samples]
    from torch.utils.data import Subset
    return Subset(dataset, indices)

# Training function with VAE loss
def train_vae_model(model, train_loader, epochs, lr, device, beta=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        batch_count = 0
        
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            reconstructed, _, mu, logvar = model(imgs)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, imgs)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
            
            # Total loss
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            total_recon += recon_loss.item() * imgs.size(0)
            total_kl += kl_loss.item() * imgs.size(0)
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = total_loss / len(train_loader.dataset)
            avg_recon = total_recon / len(train_loader.dataset)
            avg_kl = total_kl / len(train_loader.dataset)
            
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
        
        scheduler.step()
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}) - LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Update loss chart
        if len(train_losses) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_losses, label='Total Loss', color='#6c5ce7', linewidth=2)
            ax.plot(recon_losses, label='Reconstruction Loss', color='#00b894', linewidth=2)
            ax.plot(kl_losses, label='KL Loss', color='#fd79a8', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            loss_chart.pyplot(fig)
            plt.close(fig)
    
    return train_losses, recon_losses, kl_losses

# Compression utilities
def compress_image(model, image_tensor, device):
    """Compress image to quantized soul seed bytes"""
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(image_tensor.unsqueeze(0).to(device))
        mu = encoded[:, :model.soul_dim]
        
        # Quantize to 8-bit
        soul_seed = mu.cpu().squeeze()
        quantized = torch.clamp(torch.round(soul_seed * 127), -128, 127).to(torch.int8)
        
        # Convert to bytes
        compressed_bytes = quantized.numpy().tobytes()
        return compressed_bytes, len(compressed_bytes)

def decompress_bytes(model, compressed_bytes, device):
    """Decompress from bytes to image"""
    model.eval()
    with torch.no_grad():
        # Convert bytes back to tensor
        quantized = torch.tensor(np.frombuffer(compressed_bytes, dtype=np.int8), dtype=torch.float32)
        soul_seed = (quantized / 127.0).unsqueeze(0).to(device)
        
        # Decode
        reconstructed = model.decoder(soul_seed)
        return reconstructed.squeeze().cpu()

# Main app
def main():
    st.markdown('<h1 class="main-header">📸 SoulGenesis 128 - Real Image Compression</h1>', unsafe_allow_html=True)
    
    st.sidebar.header("Configuration")
    dataset_name = st.sidebar.selectbox("Dataset", ["cifar10", "stl10"], index=0)
    soul_dim = st.sidebar.slider("Soul Seed Dimension", 256, 1024, 512)
    epochs = st.sidebar.slider("Training Epochs", 5, 30, 15)
    batch_size = st.sidebar.slider("Batch Size", 8, 32, 16)
    learning_rate = st.sidebar.slider("Learning Rate", 1e-4, 1e-2, 5e-4, format="%.4f")
    num_samples = st.sidebar.slider("Training Samples", 500, 5000, 1000)
    beta = st.sidebar.slider("KL Weight (β)", 0.001, 0.5, 0.01, format="%.3f")
    
    device = "cpu"
    st.sidebar.info(f"Using {device.upper()} - Training real 128x128 images")
    
    # Load real dataset
    with st.spinner("Loading real image dataset..."):
        dataset = load_real_dataset(dataset_name, img_size=128, num_samples=num_samples)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize VAE model
    model = VAESoulGenesis(soul_dim=soul_dim).to(device)
    
    # Model info
    param_count = sum(p.numel() for p in model.parameters())
    st.sidebar.info(f"Model parameters: {param_count:,}")
    
    # Compression info
    original_size = 3 * 128 * 128  # 49,152 values for 128x128 RGB
    compressed_size = soul_dim  # soul seed dimension
    compression_ratio = original_size / compressed_size
    theoretical_ratio = original_size / (compressed_size * 0.125)  # 8-bit quantization
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Image Size", "128x128 RGB")
        st.metric("Original Values", f"{original_size:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Soul Seed Size", f"{compressed_size} values")
        st.metric("Theoretical CR", f"{theoretical_ratio:.1f}x")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Dataset", dataset_name.upper())
        st.metric("Samples", f"{num_samples:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.metric("Target Error", "< 0.02")
        st.metric("VAE Mode", "✓ Enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show sample images from real dataset
    st.markdown('<div class="sub-header">🎨 Real Dataset Samples</div>', unsafe_allow_html=True)
    sample_indices = np.random.choice(len(dataset), 6, replace=False)
    cols = st.columns(6)
    for i, col in enumerate(cols):
        with col:
            img, label = dataset[sample_indices[i]]
            st.image(img.permute(1, 2, 0).numpy(), use_column_width=True)
            st.caption(f"Sample {i+1} (Class: {label})")
    
    # Training section
    if st.button("🚀 Train on Real Images", type="primary"):
        st.markdown('<div class="sub-header">📈 VAE Training Progress</div>', unsafe_allow_html=True)
        
        with st.spinner("Training VAE on real 128x128 images..."):
            train_losses, recon_losses, kl_losses = train_vae_model(
                model, train_loader, epochs, learning_rate, device, beta
            )
        
        # Results analysis
        st.markdown('<div class="sub-header">🎯 Reconstruction Results</div>', unsafe_allow_html=True)
        
        model.eval()
        with torch.no_grad():
            # Get test batch
            test_batch = next(iter(train_loader))
            imgs, labels = test_batch
            imgs = imgs.to(device)
            
            # Reconstruct
            recons, souls, _, _ = model(imgs[:6])
            
            # Display results
            st.write("**Original (top) vs Reconstructed (bottom):**")
            cols = st.columns(6)
            
            total_psnr = 0
            total_ssim = 0
            
            for i in range(6):
                with cols[i]:
                    # Original
                    orig_img = imgs[i].cpu().permute(1, 2, 0).numpy()
                    st.image(orig_img, use_column_width=True, caption=f"Original {i+1}")
                    
                    # Reconstructed
                    recon_img = recons[i].cpu().permute(1, 2, 0).numpy()
                    st.image(recon_img, use_column_width=True, caption=f"Reconstructed {i+1}")
                    
                    # Calculate metrics
                    mse = F.mse_loss(recons[i], imgs[i]).item()
                    psnr = 10 * torch.log10(1.0 / torch.tensor(mse)).item() if mse > 0 else 100
                    total_psnr += psnr
                    
                    st.caption(f"PSNR: {psnr:.1f} dB")
            
            avg_psnr = total_psnr / 6
            st.success(f"✅ Average PSNR: {avg_psnr:.1f} dB (Higher is better)")
            
            if avg_psnr > 30: 