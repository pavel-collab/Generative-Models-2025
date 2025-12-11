import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from modules import DDPM, DDPMConfig, CFG

plt.style.use("ggplot")

cifar_dataset = CIFAR10(
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

# Функция обучения DDPM
def train_ddpm():
    """
    Основной цикл обучения DDPM
    """
    
    # Конфигурация
    config = DDPMConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Создаем модель
    ddpm = DDPM(config).to(device)

    # Оптимизатор
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=config.lr)

    # DataLoader
    train_loader = DataLoader(
        cifar_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Для сохранения результатов
    sample_dir = "./ddpm_images/"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # История обучения
    losses = []

    print("Начало обучения DDPM...")
    print(f"Устройство: {device}")
    print(f"Количество эпох: {config.num_epochs}")
    print(f"Размер батча: {config.batch_size}")
    print(f"Количество временных шагов: {config.timesteps}")
    print("-" * 80)

    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_losses = []

        # Прогресс бар для эпохи
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')

        for _, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            print(f"[DEBUG] model device {next(ddpm.parameters()).device}")
            print(f"[DEBUG] image device {images.device}")

            # Прямой проход и вычисление loss
            optimizer.zero_grad()
            loss = ddpm(images)


            # Обратное распространение
            loss.backward()

            # Gradient clipping для стабильност
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            
            optimizer.step()

            # Сохраняем loss
            epoch_losses.append(loss.item())
            losses.append(loss.item())

            # Обновляем прогресс бар
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{config.num_epochs} - "
              f"Avg Loss: {avg_loss:.4f} - "
              f"Time: {epoch_time:.1f}s")

        # Генерация примеров каждые 10 эпох
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Генерация примеров для эпохи {epoch+1}...")
            
            samples = ddpm.sample(batch_size=64, device=device)
            samples = (samples + 1) / 2  # Денормализация из [-1, 1] в [0, 1]
            samples = torch.clamp(samples, 0, 1)

            save_path = os.path.join(sample_dir, f'samples_epoch_{epoch+1:04d}.png')
            save_image(samples, save_path, nrow=8)

            print(f"Сохранено: {save_path}")

        # Сохранение чекпоинта каждые 50 эпох
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(sample_dir, f'ddpm_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"Чекпоинт сохранен: {checkpoint_path}")

        print("-" * 80)

    total_time = time.time() - start_time

    print(f"Обучение DDPM завершено! Общее время: {total_time/60:.1f} минут")
    
    return ddpm, losses

# Функция для визуализации результатов обучения
def plot_ddpm_training(losses, save_path='./images/'):
    """
    Визуализация кривой обучения DDPM
    """

    plt.figure(figsize=(12, 5))

    # График всех потерь
    plt.subplot(1, 2, 1)
    plt.plot(losses, alpha=0.6, linewidth=0.5)
    plt.xlabel('Итерация')
    plt.ylabel('Loss (MSE)')
    plt.title('DDPM Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Сглаженный график
    plt.subplot(1, 2, 2)

    window = 100
    
    if len(losses) > window:
        losses_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(losses_smooth, linewidth=2)
        plt.xlabel('Итерация (сглаженная)')
        plt.ylabel('Loss (MSE)')
        plt.title('DDPM Training Loss (Сглаженная)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ddpm_training_curve.png'), dpi=150)
    plt.show()

# Функция для генерации дополнительных примеров
def generate_samples(ddpm, num_samples=100, device='cuda', save_path='./ddpm_images/'):
    """
    Генерация набора примеров из обученной DDPM модели
    """
    print(f"Генерация {num_samples} примеров...")
    ddpm.eval()

    all_samples = []
    batch_size = 25

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            samples = ddpm.sample(batch_size=current_batch, device=device)
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            all_samples.append(samples)
            print(f"Сгенерировано {i + current_batch}/{num_samples}")

    all_samples = torch.cat(all_samples, dim=0)

    # Сохраняем как одно большое изображение
    save_image(all_samples, os.path.join(save_path, 'ddpm_final_samples.png'), nrow=10)
    print(f"Все примеры сохранены в {save_path}")

    return all_samples

# Обучаем модель
ddpm_model, losses = train_ddpm()

# Визуализируем результаты обучения
plot_ddpm_training(losses)

# Генерируем финальные примеры
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_samples = generate_samples(ddpm_model, num_samples=100, device=device)

print("\nОбучение и генерация DDPM завершены!")

MODEL_SAVE_PATH = './models'
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

torch.save(ddpm_model.state_dict(), f'{MODEL_SAVE_PATH}/ddpm.pth')
