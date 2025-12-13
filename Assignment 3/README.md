Подготовка к запуску
```
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Установка пакетов для подстчета метрик
```
git clone https://github.com/w86763777/pytorch-gan-metrics.git
pip install pytorch-gan-metrics
```

Обучение модели GAN
```
python3 train_gan.py
```

Обучение модели DDPM
```
python3 train_ddpm.py
```

Проверка тестами
```
cd tests
pytest model_generation_test.py
```

```
cd tests
pytest model_import_test.py
```

После тренировки чекпоинты моделей и примеры сгенерированных картинок будут сохранены локально.

Сравнение метрик
```
python3 ./evaludate.py
```