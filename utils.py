import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import hsv_to_rgb
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA



def compute_receptive_fields_from_imgs(model, imgs, scaler=None, scaler_type='minmax', 
                                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compute the receptive fields (activation profiles) of each hidden neuron
    as a function of viewing angle, starting from raw images.
    
    Parameters:
    model: trained autoencoder model
    imgs: numpy array of shape (N, Nx, Ny, 3) - raw images
    scaler: fitted scaler object (if None, will create new one)
    scaler_type: type of scaling to apply
    
    Returns:
    receptive_fields: numpy array of shape (N_images, N_hidden_neurons)
    angles: numpy array of angles corresponding to each image
    """
  
    
    model.eval()
    N_images = imgs.shape[0]
    
    # Prepare data (same as in the training function)
    imgs_flat = imgs.reshape(N_images, -1).astype(np.float32)
    
    # Apply scaling
    if scaler is None:
        # Create new scaler
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
            imgs_scaled = scaler.fit_transform(imgs_flat)
        elif scaler_type == 'standard':
            scaler = StandardScaler()
            imgs_scaled = scaler.fit_transform(imgs_flat)
        elif scaler_type == 'robust':
            scaler = RobustScaler()
            imgs_scaled = scaler.fit_transform(imgs_flat)
        elif scaler_type == 'simple':
            if imgs.dtype == np.uint8:
                imgs_scaled = imgs_flat / 255.0
            else:
                imgs_scaled = imgs_flat
        else:  # None or no scaling
            imgs_scaled = imgs_flat
    else:
        # Use provided scaler
        if scaler_type == 'simple':
            if imgs.dtype == np.uint8:
                imgs_scaled = imgs_flat / 255.0
            else:
                imgs_scaled = imgs_flat
        elif scaler_type is None:
            imgs_scaled = imgs_flat
        else:
            imgs_scaled = scaler.transform(imgs_flat)
    
    # Convert to tensor
    imgs_tensor = torch.FloatTensor(imgs_scaled).to(device)
    
    # Get hidden activations for all images
    with torch.no_grad():
        _, hidden_activations = model(imgs_tensor)
        hidden_activations = hidden_activations.cpu().numpy()
    
    # Create angle array (0 to 2π)
    angles = np.linspace(0, 2*np.pi, N_images, endpoint=False)
    
    # print(f"Computed receptive fields:")
    # print(f"  Input images shape: {imgs.shape}")
    # print(f"  Scaled data range: [{imgs_scaled.min():.3f}, {imgs_scaled.max():.3f}]")
    # print(f"  Hidden activations shape: {hidden_activations.shape}")
    # print(f"  Angles range: [0, 2π] with {N_images} steps")
    
    return hidden_activations, angles


def plot_neuron_rfs_simple(hidden_activations, angles, neuron_indices=None, 
                          figsize=(5, 4), title_suffix="", savefigfile=None, tag = ''):
    """
    Simple plot of neuron receptive fields as activation vs angle.
    
    Parameters:
    hidden_activations: array of shape (N_images, N_hidden_neurons) - from previous function
    angles: array of angles (in radians) - from previous function
    neuron_indices: list of neuron indices to plot (if None, plot all)
    figsize: figure size
    title_suffix: string to add to title
    """
    
    N_images, N_neurons = hidden_activations.shape
    angles_deg = angles * 180 / np.pi  # Convert to degrees for plotting
    
    # Determine which neurons to plot
    if neuron_indices is None:
        neuron_indices = list(range(N_neurons))
    else:
        # Ensure indices are valid
        neuron_indices = [idx for idx in neuron_indices if 0 <= idx < N_neurons]
    
    n_neurons_to_plot = len(neuron_indices)
    
    # Plot all specified RFs
    plt.figure(figsize=figsize)
    
    # Use discrete tab colors - combine tab20 with tab10 for more variety
    tab20_colors = plt.cm.tab20.colors
    tab10_colors = plt.cm.tab10.colors
    
    # Create extended discrete color palette
    discrete_colors = list(tab20_colors) + list(tab10_colors)  # 30 colors total
    
    # If we need more colors, cycle through them
    def get_color(i):
        return discrete_colors[i % len(discrete_colors)]
    
    for i, neuron_idx in enumerate(neuron_indices):
        activations = hidden_activations[:, neuron_idx]
        color = get_color(i)
        
        plt.plot(angles_deg, activations, color=color, 
                alpha=1.0, linewidth=2, label=f'Neuron {neuron_idx}')
    
    plt.xlabel('Angle', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.title(f'Receptive Fields - {n_neurons_to_plot} (active) Neurons {title_suffix}' + tag, fontsize=14)
    plt.grid(True, alpha=0.3)
    #plt.xlim(0, 360)
    
    # Set x-ticks at regular intervals
    xticks = [0, 90, 180, 270, 360]
    xtickslabels = ['0', 'π/2', 'π', '3π/2', '2π']
    #plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
   
    plt.xticks(xticks, xtickslabels, fontsize=10)
    
    # Add legend only if not too many neurons
    if n_neurons_to_plot <= 25:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    elif n_neurons_to_plot <= 50:
        # For medium number of neurons, put legend in multiple columns
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=2)
    
    # Add secondary x-axis with radians
    # ax2 = plt.gca().twiny()
    # ax2.set_xlim(0, 2*np.pi)
    # ax2.set_xlabel('Angle (radians)', fontsize=12)
    # ax2.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
    # ax2.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π'])

    
    plt.tight_layout()
    if savefigfile is not None:
        plt.savefig(savefigfile, dpi=300)
        #print(f"Saved figure to {savefigfile}")
    plt.show()
    
    # print(f"Plotted {n_neurons_to_plot} neuron receptive fields using discrete tab colors")
    # if neuron_indices != list(range(N_neurons)):
    #     print(f"Plotted neurons: {neuron_indices}")
    # return 





class TwoLayerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, 
                 l1_weight_reg=0.0, l2_weight_reg=0.0, 
                 l1_activation_reg=0.0, l2_activation_reg=0.0):
        super(TwoLayerAutoencoder, self).__init__()
        
        self.l1_weight_reg = l1_weight_reg
        self.l2_weight_reg = l2_weight_reg
        self.l1_activation_reg = l1_activation_reg
        self.l2_activation_reg = l2_activation_reg
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=use_bias)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def encode(self, x):
        return self.relu(self.encoder(x))
    
    def decode(self, h):
        return self.decoder(h)
    
    def forward(self, x):
        h = self.encode(x)
        x_reconstructed = self.decode(h)
        return x_reconstructed, h
    
    def regularization_loss(self, hidden_activations):
        reg_loss = 0.0
        
        # Weight regularization
        if self.l1_weight_reg > 0:
            l1_weight_loss = (torch.abs(self.encoder.weight).sum() + 
                            torch.abs(self.decoder.weight).sum())
            reg_loss += self.l1_weight_reg * l1_weight_loss
            
        if self.l2_weight_reg > 0:
            l2_weight_loss = (torch.norm(self.encoder.weight, 2) + 
                            torch.norm(self.decoder.weight, 2))
            reg_loss += self.l2_weight_reg * l2_weight_loss
        
        # Activation regularization
        if self.l1_activation_reg > 0:
            l1_activation_loss = torch.abs(hidden_activations).sum()
            reg_loss += self.l1_activation_reg * l1_activation_loss
            
        if self.l2_activation_reg > 0:
            l2_activation_loss = torch.norm(hidden_activations, 2)
            reg_loss += self.l2_activation_reg * l2_activation_loss
            
        return reg_loss

def prepare_data_with_scaling(imgs, scaler_type='minmax'):
    """
    Prepare and scale data with different scaling options
    
    Parameters:
    imgs: numpy array of shape (N, Nx, Ny, channels)
    scaler_type: str - 'minmax', 'standard', 'robust', 'simple', or None
    
    Returns:
    imgs_scaled: scaled flattened images
    scaler: fitted scaler object (for inverse transform)
    original_shape: original image shape for reconstruction
    """
    N, Nx, Ny, channels = imgs.shape
    original_shape = (Nx, Ny, channels)
    
    # Flatten images
    imgs_flat = imgs.reshape(N, -1).astype(np.float32)
    
    if scaler_type is None:
        # No scaling
        imgs_scaled = imgs_flat
        scaler = None
    elif scaler_type == 'simple':
        # Simple normalization (divide by 255 if uint8)
        if imgs.dtype == np.uint8:
            imgs_scaled = imgs_flat / 255.0
        else:
            imgs_scaled = imgs_flat
        scaler = None
    elif scaler_type == 'minmax':
        # Min-Max scaling to [0, 1]
        scaler = MinMaxScaler()
        imgs_scaled = scaler.fit_transform(imgs_flat)
    elif scaler_type == 'standard':
        # Standard scaling (zero mean, unit variance)
        scaler = StandardScaler()
        imgs_scaled = scaler.fit_transform(imgs_flat)
    elif scaler_type == 'robust':
        # Robust scaling (uses median and IQR)
        scaler = RobustScaler()
        imgs_scaled = scaler.fit_transform(imgs_flat)
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    print(f"Data scaling: {scaler_type}")
    print(f"Original data range: [{imgs_flat.min():.3f}, {imgs_flat.max():.3f}]")
    print(f"Scaled data range: [{imgs_scaled.min():.3f}, {imgs_scaled.max():.3f}]")
    print(f"Scaled data mean: {imgs_scaled.mean():.3f}, std: {imgs_scaled.std():.3f}")
    print()
    
    return imgs_scaled, scaler, original_shape

def train_autoencoder(imgs, model = None, hidden_dim=32, epochs=1000, batch_size=32, 
                     learning_rate=0.001, use_bias=True,
                     l1_weight_reg=0.0, l2_weight_reg=0.0, 
                     l1_activation_reg=0.0, l2_activation_reg=0.0,
                     scaler_type='minmax',
                     device='cuda' if torch.cuda.is_available() else 'cpu', synnoise = 0.0):
    
    # Prepare and scale data
    imgs_scaled, scaler, original_shape = prepare_data_with_scaling(imgs, scaler_type)
    input_dim = imgs_scaled.shape[1]
    
    # Convert to PyTorch tensors
    imgs_tensor = torch.FloatTensor(imgs_scaled).to(device)
    
    # Create data loader
    dataset = TensorDataset(imgs_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    if model is None:
        model = TwoLayerAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_bias=use_bias,
            l1_weight_reg=l1_weight_reg,
            l2_weight_reg=l2_weight_reg,
            l1_activation_reg=l1_activation_reg,
            l2_activation_reg=l2_activation_reg
        ).to(device)
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch_imgs = batch[0]
            
            # Forward pass
            reconstructed, hidden = model(batch_imgs)
            
            # Compute losses
            reconstruction_loss = criterion(reconstructed, batch_imgs)
            reg_loss = model.regularization_loss(hidden)
            total_loss = reconstruction_loss + reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            if synnoise>0.0:
                # Add synaptic noise to weights after gradient update
                with torch.no_grad():
                    for param in model.parameters():
                        # only weights, not biases
                        if param.ndim >= 2:
                            noise = torch.randn_like(param) * synnoise * np.sqrt(dataloader.batch_size) * np.sqrt(learning_rate)
                            param.add_(noise)

            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return model, losses, imgs_tensor, imgs_scaled, scaler, original_shape

def inverse_transform_image(img_flat, scaler, original_shape, scaler_type):
    """
    Convert scaled flattened image back to original format
    """
    if scaler is not None:
        img_unscaled = scaler.inverse_transform(img_flat.reshape(1, -1)).flatten()
    elif scaler_type == 'simple':
        img_unscaled = img_flat * 255.0
    else:
        img_unscaled = img_flat
    
    # Reshape and clip to valid range
    img_reshaped = img_unscaled.reshape(original_shape)
    
    if scaler_type == 'simple' or scaler_type == 'minmax' or scaler is None:
        img_reshaped = np.clip(img_reshaped, 0, 255 if scaler_type == 'simple' else 1)
    
    return img_reshaped

def visualize_results(model, losses, imgs_tensor, imgs_scaled, scaler, original_shape, 
                     scaler_type, imgs, 
                     device='cuda' if torch.cuda.is_available() else 'cpu', savefigfile = None):
    
    N = imgs.shape[0]
    
    # Plot training loss
    plt.figure(figsize=(15, 12))
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Make the drawing area of this subplot square
    ax1.set_box_aspect(1)   # height/width = 1
    
    # Get hidden representations
    model.eval()
    with torch.no_grad():
        _, hidden_reps = model(imgs_tensor)
        hidden_reps = hidden_reps.cpu().numpy()
        
        # Reconstruct images
        reconstructed_imgs, _ = model(imgs_tensor)
        reconstructed_imgs = reconstructed_imgs.cpu().numpy()
    
    # Show original vs reconstructed example (last image)
    original_scaled = imgs_scaled[-1]
    reconstructed_scaled = reconstructed_imgs[-1]
    
    # Convert back to original scale for visualization
    original_img = inverse_transform_image(original_scaled, scaler, original_shape, scaler_type)
    reconstructed_img = inverse_transform_image(reconstructed_scaled, scaler, original_shape, scaler_type)
    
    # Normalize for display if needed
    if scaler_type == 'simple' or scaler_type is None:
        if original_img.max() > 1:
            original_display = original_img / 255.0
            reconstructed_display = reconstructed_img / 255.0
        else:
            original_display = original_img
            reconstructed_display = reconstructed_img
    else:
        # For other scalers, normalize to [0, 1] for display
        original_display = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        reconstructed_display = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.clip(original_display, 0, 1), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.clip(reconstructed_display, 0, 1), cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    # PCA on hidden representations
    if hidden_reps.shape[1] >= 2:
        # pca = PCA(n_components=2)
        # hidden_2d = pca.fit_transform(hidden_reps)
        
        # # Create phase colors (index mapped to 0-2π)
        # phases = np.linspace(0, 2*np.pi, N, endpoint=False)
        # colors = np.zeros((N, 3))
        
        # # Convert phase to HSV then RGB
        # for i, phase in enumerate(phases):
        #     hue = phase / (2 * np.pi)  # Map to [0, 1]
        #     hsv_color = np.array([hue, 1.0, 1.0])  # Full saturation and value
        #     colors[i] = hsv_to_rgb(hsv_color.reshape(1, 1, 3)).flatten()
        
        # plt.subplot(2, 3, 4)
        # scatter = plt.scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=colors, s=50)
        # plt.title(f'PCA of Hidden Representations\n(Colored by Phase, Scaler: {scaler_type})')
        # plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
        # plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
        # plt.grid(True, alpha=0.3)


        # Add parameter to control 2D vs 3D plotting
        plot_3d = True  # Set to False for 2D plot

        # Configure PCA based on plotting preference
        n_components = 3 if plot_3d else 2
        pca = PCA(n_components=n_components)
        hidden_transformed = pca.fit_transform(hidden_reps)

        # Create phase colors (index mapped to 0-2π)
        phases = np.linspace(0, 2*np.pi, N, endpoint=False)
        colors = np.zeros((N, 3))

        # Convert phase to HSV then RGB
        for i, phase in enumerate(phases):
            hue = phase / (2 * np.pi)  # Map to [0, 1]
            hsv_color = np.array([hue, 1.0, 1.0])  # Full saturation and value
            colors[i] = hsv_to_rgb(hsv_color.reshape(1, 1, 3)).flatten()

        # Plot based on selected dimensionality
        if plot_3d:
            # 3D Plot
            ax = plt.subplot(2, 3, 4, projection='3d')
            scatter = ax.scatter(hidden_transformed[:, 0], 
                                hidden_transformed[:, 1], 
                                hidden_transformed[:, 2], 
                                c=colors, s=50)
            
            ax.set_title(f'3D PCA of Hidden Representations\n(Colored by Phase, Scaler: {scaler_type})')
            ax.set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
            ax.set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
            ax.set_zlabel(f'PC3 (Explained Variance: {pca.explained_variance_ratio_[2]:.3f})')
            
            # Optional: Add grid
            ax.grid(True, alpha=0.3)
            
        else:
            # 2D Plot (original)
            plt.subplot(2, 3, 4)
            scatter = plt.scatter(hidden_transformed[:, 0], hidden_transformed[:, 1], c=colors, s=50)
            plt.title(f'2D PCA of Hidden Representations\n(Colored by Phase, Scaler: {scaler_type})')
            plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
            plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
            plt.grid(True, alpha=0.3)
        
        #plt.subplot(2, 3, 4)
        #plt.colorbar(label='Phase (radians)')
        # Add colorbar showing phase
        ax = plt.subplot(2, 3, 5)

        scatter = ax.scatter(hidden_transformed[:, 0],
                            hidden_transformed[:, 1],
                            c=colors, s=50)
        ax.set_title(f'2D PCA of Hidden Representations\n(Colored by Phase, Scaler: {scaler_type})')
        ax.set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
        ax.set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
        ax.grid(True, alpha=0.3)

        # Make this subplot square
        ax.set_box_aspect(1)

        # Add colorbar showing phase
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
        cmap = plt.cm.hsv

        # Dummy mappable for the colorbar
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # required for older Matplotlib versions

        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Phase (radians)')

        # Optional: nice angular tick labels
        tick_locs = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(
            [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
        )
        
        
        # Plot explained variance
        ax6 = plt.subplot(2, 3, 6)

        if hidden_reps.shape[1] > 2:
            pca_full = PCA()
            pca_full.fit(hidden_reps)
            ax6.plot(np.cumsum(pca_full.explained_variance_ratio_))
            ax6.set_title('Cumulative Explained Variance')
            ax6.set_xlabel('Principal Component')
            ax6.set_ylabel('Cumulative Explained Variance Ratio')
            ax6.grid(True)
        else:
            ax6.bar(['PC1', 'PC2'], pca.explained_variance_ratio_)
            ax6.set_title('Explained Variance Ratio')
            ax6.set_ylabel('Explained Variance Ratio')
            ax6.grid(True, alpha=0.3)

        # Make this subplot square
        ax6.set_box_aspect(1)
    
    else:
        plt.subplot(2, 3, 4)
        plt.text(0.5, 0.5, 'Hidden dimension < 2\nCannot perform PCA', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('PCA Not Possible')
    
    plt.tight_layout()
    #plt.show()
    if savefigfile is not None:
        plt.savefig(savefigfile, dpi=300)
        print(f"Saved figure to {savefigfile}")
    plt.show()
    return hidden_reps, phases, colors

# Main execution function
def main_autoencoder_analysis(imgs, model = None, hidden_dim=32, epochs=500, batch_size=32, 
                             learning_rate=0.001, use_bias=True,
                             l1_weight_reg=0.0, l2_weight_reg=1e-5, 
                             l1_activation_reg=0.0, l2_activation_reg=1e-4,
                             scaler_type='minmax', visualize_results_flag=True, synnoise = 0.0, savefigfile = None):
    """
    Main function to train autoencoder and visualize results
    
    Parameters:
    imgs: numpy array of shape (N, Nx, Ny, 3) - your image dataset
    model: PyTorch model - pre-trained autoencoder model (optional)
    hidden_dim: int - number of hidden dimensions
    epochs: int - number of training epochs
    batch_size: int - batch size for training
    learning_rate: float - learning rate for optimizer
    use_bias: bool - whether to use bias in linear layers
    l1_weight_reg: float - L1 regularization coefficient for weights
    l2_weight_reg: float - L2 regularization coefficient for weights
    l1_activation_reg: float - L1 regularization coefficient for activations
    l2_activation_reg: float - L2 regularization coefficient for activations
    scaler_type: str - 'minmax', 'standard', 'robust', 'simple', or None
    """
    
    print(f"Training autoencoder with:")
    print(f"  Hidden dimensions: {hidden_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Use bias: {use_bias}")
    print(f"  Scaler type: {scaler_type}")
    print(f"  L1 weight reg: {l1_weight_reg}")
    print(f"  L2 weight reg: {l2_weight_reg}")
    print(f"  L1 activation reg: {l1_activation_reg}")
    print(f"  L2 activation reg: {l2_activation_reg}")
    print()
    
    # Train the autoencoder
    model, losses, imgs_tensor, imgs_scaled, scaler, original_shape = train_autoencoder(
        imgs, model = model, hidden_dim=hidden_dim, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, use_bias=use_bias,
        l1_weight_reg=l1_weight_reg, l2_weight_reg=l2_weight_reg,
        l1_activation_reg=l1_activation_reg, l2_activation_reg=l2_activation_reg,
        scaler_type=scaler_type, synnoise=synnoise
    )
    
    # Visualize results
    if visualize_results_flag:
        hidden_reps, phases, colors = visualize_results(
            model, losses, imgs_tensor, imgs_scaled, scaler, original_shape, 
            scaler_type, imgs, savefigfile=savefigfile
        )
    else:
        hidden_reps, phases, colors = None, None, None
    
    return model, hidden_reps, phases, colors, scaler



def run_model(model = None, imgs = None, params = None):
    model, hidden_reps, phases, colors, scaler = main_autoencoder_analysis(
        imgs, 
        model = model,
        hidden_dim=params['hidden_dim'],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        use_bias=params['use_bias'],
        l1_weight_reg=params['l1_weight_reg'],
        l2_weight_reg=params['l2_weight_reg'],
        l1_activation_reg=params['l1_activation_reg'],
        l2_activation_reg=params['l2_activation_reg'],
        visualize_results_flag=params['visualize_results_flag'],
        synnoise = params['synnoise'],
        savefigfile=params['savefigfile']
    )
    return model, hidden_reps, phases, colors, scaler




def compute_spectra(signals, fs=1.0):
    """
    Compute single-sided magnitude and phase spectra for one or more 1D signals.

    Parameters
    ----------
    signals : array-like
        1D array (n_samples,) or 2D array (n_signals, n_samples).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    freqs_pos : np.ndarray, shape (n_freqs,)
        Non-negative frequency bins.
    mags : np.ndarray, shape (n_signals, n_freqs)
        Magnitude spectra.
    phases : np.ndarray, shape (n_signals, n_freqs)
        Phase spectra (radians).
    """
    signals = np.asarray(signals)

    # Allow single signal or multiple
    if signals.ndim == 1:
        signals = signals[None, :]  # (1, N)

    assert signals.ndim == 2, "signals must be 1D or 2D: (n_samples,) or (n_signals, n_samples)"

    n_signals, N = signals.shape

    # Frequency axis (same for all signals)
    freqs = np.fft.fftfreq(N, d=1.0/fs)
    idx = freqs >= 0
    freqs_pos = freqs[idx]

    mags_all = []
    phases_all = []

    for i in range(n_signals):
        X = np.fft.fft(signals[i])
        mags_all.append(np.abs(X[idx]))
        phases_all.append(np.angle(X[idx]))

    mags_all = np.array(mags_all)
    phases_all = np.array(phases_all)

    return freqs_pos, mags_all, phases_all



def get_reconstructed_imgs(model,
                           imgs,
                           scaler=None,
                           scaler_type='minmax',
                           device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Run imgs through the autoencoder and return reconstructed images
    in the original image space and shape, ready for display.

    Parameters
    ----------
    model : trained TwoLayerAutoencoder
    imgs : np.ndarray, shape (N, Nx, Ny, C)
        Original images.
    scaler : fitted scaler or None
        Same scaler returned by training (MinMaxScaler, StandardScaler, etc.).
    scaler_type : str
        Same string used in training: 'minmax', 'standard', 'robust', 'simple', or None.
    device : 'cuda' or 'cpu'

    Returns
    -------
    recon_imgs : np.ndarray, shape (N, Nx, Ny, C)
        Reconstructed images in original scale (e.g. 0–255 for 'simple', or
        same units as original when using a sklearn scaler).
    """
    model.eval()
    N, Nx, Ny, C = imgs.shape
    original_shape = (Nx, Ny, C)

    # Flatten and scale like in training
    imgs_flat = imgs.reshape(N, -1).astype(np.float32)

    if scaler is not None:
        imgs_scaled = scaler.transform(imgs_flat)
    elif scaler_type == 'simple':
        if imgs.dtype == np.uint8:
            imgs_scaled = imgs_flat / 255.0
        else:
            imgs_scaled = imgs_flat
    else:
        imgs_scaled = imgs_flat

    # Forward pass
    imgs_tensor = torch.from_numpy(imgs_scaled).to(device)
    with torch.no_grad():
        recon_scaled, _ = model(imgs_tensor)
    recon_scaled = recon_scaled.cpu().numpy()

    # Inverse transform back to original scale
    if scaler is not None:
        recon_unscaled = scaler.inverse_transform(recon_scaled)
    elif scaler_type == 'simple':
        recon_unscaled = recon_scaled * 255.0
    else:
        recon_unscaled = recon_scaled

    # Reshape to original image shape
    recon_imgs = recon_unscaled.reshape(N, *original_shape)

    return recon_imgs




def visualize_autoencoder_results(model,
                                       imgs,
                                       scaler,
                                       scaler_type,
                                       device='cuda' if torch.cuda.is_available() else 'cpu',
                                       savefigfile=None):
    """
    Visualization (1 x 3):
      - original image (one example)
      - reconstructed image (same example)
      - PCA of hidden reps (3D if possible, otherwise 2D)

    Parameters
    ----------
    model : trained TwoLayerAutoencoder
    imgs : np.ndarray, shape (N, Nx, Ny, C)
    scaler : fitted scaler or None (as returned by training)
    scaler_type : str, same used in training ('minmax', 'standard', 'robust', 'simple', or None)
    device : str, 'cuda' or 'cpu'
    savefigfile : str or None, path to save the figure (if not None)
    """
    import matplotlib as mpl
    model.eval()
    N, Nx, Ny, C = imgs.shape
    original_shape = (Nx, Ny, C)

    # Flatten + scale data in the same way as train_autoencoder
    imgs_flat = imgs.reshape(N, -1).astype(np.float32)
    if scaler is not None:
        imgs_scaled = scaler.transform(imgs_flat)
    elif scaler_type == 'simple':
        if imgs.dtype == np.uint8:
            imgs_scaled = imgs_flat / 255.0
        else:
            imgs_scaled = imgs_flat
    else:
        imgs_scaled = imgs_flat

    # To tensor
    imgs_tensor = torch.FloatTensor(imgs_scaled).to(device)

    # Forward pass through model
    with torch.no_grad():
        reconstructed_imgs, hidden = model(imgs_tensor)
        reconstructed_imgs = reconstructed_imgs.cpu().numpy()
        hidden_reps = hidden.cpu().numpy()

    # One example: take last image
    original_scaled = imgs_scaled[-1]
    reconstructed_scaled = reconstructed_imgs[-1]

    # --- inverse scaling (same logic as inverse_transform_image) ---
    if scaler is not None:
        original_unscaled = scaler.inverse_transform(
            original_scaled.reshape(1, -1)
        ).flatten()
        reconstructed_unscaled = scaler.inverse_transform(
            reconstructed_scaled.reshape(1, -1)
        ).flatten()
    elif scaler_type == 'simple':
        original_unscaled = original_scaled * 255.0
        reconstructed_unscaled = reconstructed_scaled * 255.0
    else:
        original_unscaled = original_scaled
        reconstructed_unscaled = reconstructed_scaled

    original_img = original_unscaled.reshape(original_shape)
    reconstructed_img = reconstructed_unscaled.reshape(original_shape)

    # Normalize for display
    if scaler_type in ['simple', None]:
        if original_img.max() > 1:
            original_display = original_img / 255.0
            reconstructed_display = reconstructed_img / 255.0
        else:
            original_display = original_img
            reconstructed_display = reconstructed_img
    else:
        # per-image min-max for visualization
        def norm01(x):
            xmin, xmax = x.min(), x.max()
            if xmax > xmin:
                return (x - xmin) / (xmax - xmin)
            else:
                return np.zeros_like(x)
        original_display = norm01(original_img)
        reconstructed_display = norm01(reconstructed_img)

    # Prepare phases/colors for plotting
    phases = np.linspace(0, 2*np.pi, N, endpoint=False)
    colors = np.zeros((N, 3))
    for i, phase in enumerate(phases):
        hue = phase / (2 * np.pi)  # [0, 1]
        hsv_color = np.array([hue, 1.0, 1.0])
        colors[i] = hsv_to_rgb(hsv_color.reshape(1, 1, 3)).flatten()

    # PCA of hidden reps
    plot_3d = True
    n_components = 3 if plot_3d else 2

    if hidden_reps.shape[1] >= 2:
        pca = PCA(n_components=n_components)
        hidden_transformed = pca.fit_transform(hidden_reps)

    # ---- Plot layout: 1 x 3 (only first row) ----
    plt.figure(figsize=(15, 4))

    # (1) Original image
    ax1 = plt.subplot(1, 3, 1)
    if C == 1:
        ax1.imshow(np.clip(original_display.squeeze(-1), 0, 1), cmap='gray')
    else:
        ax1.imshow(np.clip(original_display, 0, 1))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # (2) Reconstructed image
    ax2 = plt.subplot(1, 3, 2)
    if C == 1:
        ax2.imshow(np.clip(reconstructed_display.squeeze(-1), 0, 1), cmap='gray')
    else:
        ax2.imshow(np.clip(reconstructed_display, 0, 1))
    ax2.set_title('Reconstructed Image')
    ax2.axis('off')

    # (3) PCA of hidden reps
    if hidden_reps.shape[1] >= 2:
        if plot_3d and hidden_reps.shape[1] >= 3:
            ax3 = plt.subplot(1, 3, 3, projection='3d')
            ax3.scatter(hidden_transformed[:, 0],
                        hidden_transformed[:, 1],
                        hidden_transformed[:, 2],
                        c=colors, s=50)
            ax3.set_title('3D PCA of Hidden Reps')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            ax3.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
            ax3.grid(True, alpha=0.3)
        else:
            ax3 = plt.subplot(1, 3, 3)
            ax3.scatter(hidden_transformed[:, 0],
                        hidden_transformed[:, 1],
                        c=colors, s=50)
            ax3.set_title('2D PCA of Hidden Reps')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            ax3.grid(True, alpha=0.3)
    else:
        ax3 = plt.subplot(1, 3, 3)
        ax3.text(0.5, 0.5, 'Hidden dimension < 2\nCannot perform PCA',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('PCA Not Possible')
        ax3.axis('off')

    plt.tight_layout()
    if savefigfile is not None:
        plt.savefig(savefigfile, dpi=300)
        print(f"Saved figure to {savefigfile}")
    plt.show()

    return hidden_reps, phases, colors

