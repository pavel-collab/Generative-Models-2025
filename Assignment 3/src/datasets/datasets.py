from torch.utils.data import Dataset
import torch

class GeneratorDataset(Dataset):
    """
    Датасет для генерации изображений из обученного GAN генератора
    """

    def __init__(self, G, z_dim, num_samples=5000, device='cuda'):
        self.G = G
        self.z_dim = z_dim
        self.num_samples = num_samples
        self.device = device
        self.G.eval()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        with torch.no_grad():
            # Генерируем случайный шум
            z = torch.randn(1, self.z_dim, 1, 1).to(self.device)

            # Генерируем изображение
            img = self.G(z)[0]

            # Денормализация из [-1, 1] в [0, 1]
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)

        return img

# Dataset для DDPM модели
class DDPMDataset(Dataset):
    """
    Датасет для генерации изображений из обученной DDPM модели
    """
    def __init__(self, ddpm, num_samples=5000, device='cuda'):
        self.ddpm = ddpm
        self.num_samples = num_samples
        self.device = device
        self.ddpm.eps_model.eval()

        # Предгенерируем все изображения (может занять время)
        print(f"Предгенерация {num_samples} изображений из DDPM...")

        self.images = []
        
        with torch.no_grad():
            for i in range(0, num_samples):
                samples = self.ddpm.sample()
                samples = (samples + 1) / 2
                samples = torch.clamp(samples, 0, 1)
                self.images.append(samples.cpu())
                if i % 500 == 0:
                    print(f"Сгенерировано {i}/{num_samples}")


        self.images = torch.cat(self.images, dim=0)
        print("Предгенерация завершена!")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.images[index]

# Dataset для реальных данных CIFAR10
class RealDataset(Dataset):
    """
    Датасет для реальных изображений CIFAR10 (для вычисления метрик)
    """

    def __init__(self, cifar_dataset, num_samples=5000):
        self.dataset = cifar_dataset
        self.num_samples = min(num_samples, len(cifar_dataset))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, _ = self.dataset[index]

        # Денормализация из [-1, 1] в [0, 1]
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)

        return img
