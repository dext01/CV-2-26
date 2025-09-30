# 📷 CV-2-26 — Компьютерное зрение
### 🔍 Сравнение изображений по SSIM и MSE

Этот проект сравнивает два изображения и выводит метрики:  
- **SSIM** (*Structural Similarity Index*) — индекс структурного сходства (0…1, где 1 = полное совпадение).  
- **MSE** (*Mean Squared Error*) — среднеквадратичная ошибка (0 = идентичные изображения).  

---

## 📂 Структура проекта

```
image-ssim-project/
├── compare_images.py      # основной скрипт
├── README.md              # документация проекта
├── requirements.txt       # список зависимостей
├── image1.png             # изображение №1 (пример)
├── image2.jpg             # изображение №2 (пример)
└── tests/                 # (опционально) папка для автотестов
    └── test_data_gen.py   # генерация тестовых картинок
```

---

## 🖼️ Подготовка изображений

- Положите **два файла** рядом с `compare_images.py`.  
- Названия должны быть строго:
  - `image1.<расширение>`
  - `image2.<расширение>`
- Поддерживаемые форматы: `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`.  

💡 Пример: `image1.png` + `image2.jpg`

---

## ⚙️ Установка зависимостей

### 🔹 Через `venv` (рекомендовано)

**Windows (PowerShell):**
```powershell
cd image-ssim-project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS (bash/zsh):**
```bash
cd image-ssim-project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 📦 requirements.txt

```
numpy
scikit-image
matplotlib
```

---

## 🚀 Запуск проекта

**Windows (PowerShell):**
```powershell
python .\compare_images.py
```

**Linux / macOS:**
```bash
python3 ./compare_images.py
```

Скрипт:
1. найдёт `image1.*` и `image2.*`,  
2. посчитает SSIM и MSE,  
3. выведет значения в консоль,  
4. покажет визуализацию «рядом-рядом» через Matplotlib.  

---

## ⚡ Быстрый тест (генерация картинок)

Если нет готовых файлов — сгенерируйте тестовую пару:

```python
from skimage import data, util, transform, io
import numpy as np

# Базовое изображение
img = data.camera()

# Сохраняем оригинал
io.imsave("image1.png", img)

# Делаем копию с шумом и поворотом
rot = transform.rotate(img, angle=5, resize=False, mode="edge", preserve_range=True).astype(np.uint8)
noisy = util.random_noise(rot, mode="gaussian", var=0.002)
noisy = (noisy * 255).astype(np.uint8)
io.imsave("image2.png", noisy)
```

Теперь запустите:
```bash
python compare_images.py
```

---

## ✅ Ожидаемый вывод

```
SSIM: 0.87
MSE : 120.45
Вердикт: Похожие ✅
```

🖼️ Визуализация отобразит два изображения рядом с заголовком SSIM и MSE.  

---

## ⚠️ Возможные ошибки

- **FileNotFoundError** — не найдены `image1` или `image2`.  
  ➝ Проверьте имена и расширения файлов.  

- **ModuleNotFoundError** — не найдена библиотека.  
  ➝ Установите зависимости (`pip install -r requirements.txt`).  

- **Нет окна визуализации**  
  ➝ В headless-системах (WSL/сервер) можно сохранить график в файл:  
  ```python
  plt.savefig("output.png")
  ```

---

## 📜 Лицензия

Проект распространяется под лицензией **MIT**.  
Свободно используйте и модифицируйте.  

---

✍️ *Работа выполнена в рамках задания **CV-2-26**, тема: «Компьютерное зрение».*
