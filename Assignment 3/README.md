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
pytest ./tests/model_import_test.py
```

После тренировки чекпоинты моделей и примеры сгенерированных картинок будут сохранены локально.

Для начала нужно сгенерировать статистику по исходному датасету. Для этого надо извлечь картинки из скачанного датасета, потому что при скачивании cifar10 картинки хранятся в сериализованном бинарном представлении в специальных batch файлах. Данные сериализованы с помощью библиотеки pickle. Для их извлечения используем сециальный скрипт 
```
python3 src/lib/utils/unpack_dataset.py --batch_path <path to cifar-10-batches-py>
```
Картинки будут извлечены из сериализованного представления и сохранены в формате png в каталоге data_images.
После этого, используя библиотеку pytorch_image_generation_metrics сохраняем статистику по исходному датасету в файл
```
python -m pytorch_image_generation_metrics.fid_ref \
        --path ./data_images \
        --output data/fid_ref.npz
```

Сравнение метрик
```
python3 ./evaludate.py
```
