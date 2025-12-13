import torch 
import os

# Вспомогательные функции для диффузии
def get_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Линейное расписание для beta параметров
    """

    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    """
    Извлечение значений из массива a по индексам t
    и изменение формы для broadcasting
    """

    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))