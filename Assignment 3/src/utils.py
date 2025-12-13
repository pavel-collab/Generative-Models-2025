import torch
import os
from torch.utils.data import DataLoader
from pytorch_image_generation_metrics import (get_inception_score,
                                              get_fid,
                                              get_inception_score_and_fid)

from modules.models import Generator, Discriminator
from modules.config import CFG
from datasets import GeneratorDataset, DDPMDataset, RealDataset

def import_pretrained_generator(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        generator = Generator(CFG.nc, CFG.nz, CFG.ngf)
        generator.load_state_dict(checkpoint)
        
        return generator
    else:
        print(f"Checkpoint path is incorrect: {checkpoint_path}")
        
def import_pretrained_discriminator(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        discriminator = Discriminator(CFG.nc, CFG.ndf)
        discriminator.load_state_dict(checkpoint)
        
        return discriminator
    else:
        print(f"Checkpoint path is incorrect: {checkpoint_path}")
        
# Функция для вычисления метрик для GAN
def evaluate_gan(G, z_dim, cifar_dataset, num_samples=5000, device='cuda'):
    """
    Вычисление IS и FID для GAN генератора
    """

    print("\n" + "="*80)
    print("ОЦЕНКА GAN МОДЕЛИ")
    print("="*80)

    # Создаем датасет для генератора
    gen_dataset = GeneratorDataset(G, z_dim, num_samples, device)
    gen_loader = DataLoader(gen_dataset, batch_size=50, num_workers=2)

    print(f"\nВычисление метрик на {num_samples} сгенерированных изображениях...")

    # Вычисляем IS и FID
    try:
        IS_mean, IS_std, FID = get_inception_score_and_fid(
            gen_loader, 
            './data/cifar10_train_stats.npz',  # Статистики реального датасета
            verbose=True,
            use_torch=True
        )

        print(f"\nРезультаты для GAN:")
        print(f"  Inception Score: {IS_mean:.3f} ± {IS_std:.3f}")
        print(f"  FID: {FID:.3f}")

        return {
            'IS_mean': IS_mean,
            'IS_std': IS_std,
            'FID': FID
        }

    except Exception as e:
        print(f"Ошибка при вычислении метрик: {e}")
        print("Попытка вычислить метрики по отдельности...")

        # Вычисляем IS отдельно
        IS_mean, IS_std = get_inception_score(gen_loader, use_torch=True, verbose=True)

        print(f"  Inception Score: {IS_mean:.3f} ± {IS_std:.3f}")

        return {
            'IS_mean': IS_mean,
            'IS_std': IS_std,
            'FID': None
        }

# Функция для вычисления метрик для DDPM
def evaluate_ddpm(ddpm, cifar_dataset, num_samples=5000, device='cuda'):
    """
    Вычисление IS и FID для DDPM модели
    """
    print("\n" + "="*80)
    print("ОЦЕНКА DDPM МОДЕЛИ")
    print("="*80)

    # Создаем датасет для DDPM
    ddpm_dataset = DDPMDataset(ddpm, num_samples, device)
    ddpm_loader = DataLoader(ddpm_dataset, batch_size=1, num_workers=2)

    print(f"\nВычисление метрик на {num_samples} сгенерированных изображениях...")

    # Вычисляем IS и FID
    try:
        IS_mean, IS_std, FID = get_inception_score_and_fid(
            ddpm_loader,
            './data/cifar10_train_stats.npz',
            verbose=True,
            use_torch=True
        )

        print(f"\nРезультаты для DDPM:")
        print(f"  Inception Score: {IS_mean:.3f} ± {IS_std:.3f}")
        print(f"  FID: {FID:.3f}")

        return {
            'IS_mean': IS_mean,
            'IS_std': IS_std,
            'FID': FID
        }

    except Exception as e:
        print(f"Ошибка при вычислении метрик: {e}")
        print("Попытка вычислить метрики по отдельности...")

        IS_mean, IS_std = get_inception_score(ddpm_loader, use_torch=True, verbose=True)
        print(f"  Inception Score: {IS_mean:.3f} ± {IS_std:.3f}")

        return {
            'IS_mean': IS_mean,
            'IS_std': IS_std,
            'FID': None
        }

# Функция для вычисления метрик реального датасета
def evaluate_real_dataset(cifar_dataset, num_samples=5000):
    """
    Вычисление метрик для реального датасета CIFAR10 (baseline)
    """

    print("\n" + "="*80)
    print("ОЦЕНКА РЕАЛЬНОГО ДАТАСЕТА CIFAR10 (BASELINE)")
    print("="*80)

    real_dataset = RealDataset(cifar_dataset, num_samples)
    real_loader = DataLoader(real_dataset, batch_size=50, num_workers=2)

    print(f"\nВычисление метрик на {num_samples} реальных изображениях...")

    try:
        IS_mean, IS_std = get_inception_score(real_loader, use_torch=True, verbose=True)

        print(f"\nРезультаты для реального датасета:")
        print(f"  Inception Score: {IS_mean:.3f} ± {IS_std:.3f}")
        print(f"  (FID не вычисляется для реального датасета относительно себя)")

        return {
            'IS_mean': IS_mean,
            'IS_std': IS_std,
            'FID': 0.0  # FID реального датасета относительно себя = 0
        }
    except Exception as e:
        print(f"Ошибка при вычислении метрик: {e}")
        return None

# Функция для создания сравнительной таблицы
def create_comparison_table(results_dict):
    """
    Создание таблицы сравнения результатов
    """

    import pandas as pd

    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)

    # Создаем DataFrame
    df_data = []
    for model_name, results in results_dict.items():
        if results:
            df_data.append({
                'Модель': model_name,
                'IS (mean)': f"{results['IS_mean']:.3f}",
                'IS (std)': f"{results['IS_std']:.3f}",
                'FID': f"{results['FID']:.3f}" if results['FID'] is not None else 'N/A'
            })


    df = pd.DataFrame(df_data)
    print("\n", df.to_string(index=False))

    # Сохраняем в CSV
    df.to_csv('metrics_comparison.csv', index=False)
    print("\nТаблица сохранена в 'metrics_comparison.csv'")

    # Визуализация
    import matplotlib.pyplot as plt


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График IS
    models = [r['Модель'] for r in df_data]
    is_means = [results_dict[m]['IS_mean'] for m in models]
    is_stds = [results_dict[m]['IS_std'] for m in models]

    axes[0].bar(models, is_means, yerr=is_stds, capsize=5, alpha=0.7, color=['blue', 'green', 'orange'])
    axes[0].set_ylabel('Inception Score')
    axes[0].set_title('Inception Score Comparison')
    axes[0].grid(True, alpha=0.3)

    # График FID
    fids = [results_dict[m]['FID'] for m in models if results_dict[m]['FID'] is not None]
    models_fid = [m for m in models if results_dict[m]['FID'] is not None]

    axes[1].bar(models_fid, fids, alpha=0.7, color=['blue', 'green', 'orange'])
    axes[1].set_ylabel('FID (lower is better)')
    axes[1].set_title('FID Comparison')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150)
    plt.show()

    return df

# Главная функция для комплексной оценки
def run_full_evaluation(G, z_dim, ddpm, cifar_dataset, num_samples=5000, device='cuda'):
    """
    Запуск полной оценки всех моделей
    """
    print("\n" + "="*80)
    print("ЗАПУСК КОМПЛЕКСНОЙ ОЦЕНКИ МОДЕЛЕЙ")
    print("="*80)
    print(f"Количество семплов для оценки: {num_samples}")
    print(f"Устройство: {device}")    

    results = {}

    # Оценка реального датасета
    print("\n1/3: Оценка реального датасета...")
    results['Real CIFAR10'] = evaluate_real_dataset(cifar_dataset, num_samples)

    # Оценка GAN
    print("\n2/3: Оценка GAN...")
    results['GAN'] = evaluate_gan(G, z_dim, cifar_dataset, num_samples, device)
    
    # Оценка DDPM
    print("\n3/3: Оценка DDPM...")
    results['DDPM'] = evaluate_ddpm(ddpm, cifar_dataset, num_samples, device)

    # Создаем сравнительную таблицу
    comparison_df = create_comparison_table(results)

    print("\n" + "="*80)
    print("ОЦЕНКА ЗАВЕРШЕНА!")
    print("="*80)

    return results, comparison_df
