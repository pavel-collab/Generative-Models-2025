import torch
from typing import List
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .unet import UNet
from .models import DenoiseDiffusion

class CFG:
    batch_size = 128
    num_epochs = 300
    workers = 4
    seed = 2021
    image_size = 64
    download = True
    dataroot = "data"
    nc = 3  ## number of chanels
    ngf = 64  # Size of feature maps in generator
    nz = 100  # latent random input vector
    ndf = 64  # Size of feature maps in discriminator
    lr = 0.0002
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_dir = "./images/"
    
def plot_samples(tensor):
    # Assuming you have a tensor of size torch.Size([16, 1, 32, 32])
    # Convert the tensor to a numpy array
    images = tensor.numpy()

    # Reshape the images to be of size (16, 32, 32)
    images = np.reshape(images, (16, 32, 32))

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=4)

    # Iterate over the images and plot them on the subplots
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')

    # Show the plot
    plt.show()
    
# Конфигурация для DDPM
class DDPMConfigs:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # U-Net model for epsilon_theta(x_t, t)
    eps_model: UNet
    # DDPM algorithm
    diffusion: DenoiseDiffusion

    # Number of channels in the image. 3 for RGB.
    image_channels: int = 1
    # Image size
    image_size: int = 32
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]

    # Number of time steps T
    n_steps: int = 1_000
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5

    # Number of training epochs
    epochs: int = 5

    # Dataset
    dataset: torch.utils.data.Dataset = CIFAR10(
                        root=CFG.dataroot,
                        download=CFG.download,
                        transform=transforms.Compose(
                            [
                                transforms.Resize([CFG.image_size, CFG.image_size]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]
                        ),
                    )
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam

    def init(self):
        # Create epsilon_theta(x_t, t) model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

    #! Функция для генерации изображений из гаусовского шума
    def sample(self):
        with torch.no_grad():
            # [1]
            #! создаем изображение нужной размерности из гаусовского шума
            x = torch.randn(
                self.n_samples,
                self.image_channels,
                self.image_size,
                self.image_size,
                device=self.device,
            ) # YOUR CODE HERE

            # Remove noise for T steps
            progress_bar = tqdm(range(self.n_steps))
            for t_ in progress_bar:
                progress_bar.set_description(f"Sampling")
                # t
                #! идем в обратном направлении
                t = self.n_steps - t_ - 1
                # [2]
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            plot_samples(x.detach().cpu())

    def train(self, epoch):
        # Iterate through the dataset
        progress_bar = tqdm(self.data_loader)
        for data in progress_bar:
            # Increment global step
            progress_bar.set_description(f"Epoch {epoch + 1}")
            # Move data to device
            data = data.to(self.device)

            # Make the gradients zero
            # YOUR CODE HERE
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            # YOUR CODE HERE
            loss.backward()
            # Take an optimization step
            # YOUR CODE HERE
            self.optimizer.step()
            # Track the loss
            progress_bar.set_postfix(loss=loss.detach().cpu().numpy())

    def run(self):
        for epoch in range(self.epochs):
            # Train the model
            self.train(epoch)
            # Sample some images
            self.sample()