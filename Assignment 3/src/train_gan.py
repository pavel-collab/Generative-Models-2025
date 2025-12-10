import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import os
from torch.utils.data import DataLoader

from modules import CFG, Discriminator, Generator

plt.style.use("ggplot")

if not os.path.exists(CFG.sample_dir):
    os.makedirs(CFG.sample_dir)

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

data_loader = DataLoader(
    cifar_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.workers,
    drop_last=True  # Отбрасываем последний неполный батч
)

# create new Generator model
G = Generator(CFG.nc, CFG.nz, CFG.ngf)

# create new Discriminator model
D = Discriminator(CFG.nc, CFG.ndf)

criterion = nn.BCELoss()

# Оптимизаторы Adam с параметрами из DCGAN paper
d_optimizer = torch.optim.Adam(D.parameters(), lr=CFG.lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=CFG.lr, betas=(0.5, 0.999))


def reset_grad():
    ## reset gradient for optimizer of generator and discrimator
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


LABEL_SMOOTH = 0.95
NOISE_STD = 0.1     # Стандартное отклонение для шума

def train_discriminator(images, batch_size, device):

    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones((batch_size, 1)).to(device)*LABEL_SMOOTH
    fake_labels = torch.ones((batch_size, 1)).to(device)*(1- LABEL_SMOOTH)
    
    # Это помогает дискриминатору не переобучаться
    noise = torch.randn_like(images) * NOISE_STD
    noisy_images = images + noise

    # Оценка реальных изображений
    outputs = D(noisy_images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Генерация фейковых изображений
    z = torch.randn(batch_size, CFG.nz, 1, 1).to(device)
    fake_images = G(z)

    # Добавляем шум и к фейковым изображениям
    noise_fake = torch.randn_like(fake_images) * NOISE_STD
    noisy_fake_images = fake_images + noise_fake

    # Оценка фейковых изображений
    outputs = D(noisy_fake_images.detach())  # detach() чтобы не обучать генератор
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Суммарный лосс дискриминатора
    d_loss = d_loss_real + d_loss_fake

    # Обновление весов дискриминатора
    reset_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score

def train_generator(device, batch_size):

    # Генерация латентного вектора с добавлением шума
    z = np.random.normal(0, 1, (batch_size, CFG.nz, 1, 1))
    
    # Добавляем небольшой шум для разнообразия
    noise = 0.005 * np.random.uniform() * np.abs(z).max()
    z = z + noise * np.random.normal(size=z.shape)
    z = torch.FloatTensor(z).to(device)

    # Генерация изображений
    fake_images = G(z)

    # Метки для обучения генератора (хотим обмануть дискриминатор)
    labels = torch.ones((batch_size, 1)).to(device)

    # Получаем оценку дискриминатора для сгенерированных изображений
    outputs = D(fake_images)
    g_loss = criterion(outputs, labels)

    # Обновление весов генератора
    reset_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss, fake_images

def save_fake_images(index, device, batch_size=64):
    G.eval()  # Переводим в режим evaluation

    with torch.no_grad():
        # Генерируем латентный вектор с шумом
        z = np.random.normal(0, 1, (batch_size, CFG.nz, 1, 1))
        noise = 0.005 * np.random.uniform() * np.abs(z).max()
        z = z + noise * np.random.normal(size=z.shape)
        z = torch.FloatTensor(z).to(device)

        # Генерируем изображения
        fake_images = G(z)

        # Денормализация для сохранения
        fake_images = (fake_images + 1) / 2  # Из [-1, 1] в [0, 1]

        # Сохраняем
        fake_fname = f'fake_images-{index:04d}.png'
        print(f'Saving {fake_fname}')
        save_image(fake_images, os.path.join(CFG.sample_dir, fake_fname), nrow=8)

    G.train()  # Возвращаем в режим training
    
device = CFG.device
num_epochs = CFG.num_epochs
batch_size = CFG.batch_size

total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []
G.to(device)
D.to(device)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Load a batch & transform to vectors
        images = images.to(device)
        current_batch_size = images.size(0)

        # Train the discriminator
        d_loss, real_score, fake_score = train_discriminator(
                images, current_batch_size, device
            )

        # Train the generator
        g_loss, fake_images = train_generator(device, current_batch_size)

        # Inspect the losses
        if (i+1) % 100 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))
    # Sample and save images
    save_fake_images(epoch+1, device)
    
MODEL_SAVE_PATH = './models'
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

torch.save(D.state_dict(), f'{MODEL_SAVE_PATH}/descriminator.pth')
torch.save(D.state_dict(), f'{MODEL_SAVE_PATH}/generator.pth')

import json
gan_training_dump = {
    'd_losses': d_losses,
    'g_losses': g_losses,
    'real_scores': real_scores,
    'fake_scores': fake_scores
}

JSON_DUMP_PATH = './train_info'
if not os.path.exists(JSON_DUMP_PATH):
    os.mkdir(JSON_DUMP_PATH)
    
with open(f'{JSON_DUMP_PATH}/gan_train.json', 'w') as f:
    json.dump(gan_training_dump, f)