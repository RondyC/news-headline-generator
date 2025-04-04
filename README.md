# Дообученная модель для генерации заголовков новостей

Этот репозиторий содержит код для дообучения и демонстрации модели генерации заголовков новостей на основе предобученной русскоязычной модели GPT-2. Модель была дообучена на новостном датасете с сайта [Lenta.ru](https://lenta.ru), где для каждой новости использовались столбцы с текстом и заголовком.

## Особенности

- **Модель:** Используется русскоязычная модель [sberbank-ai/rugpt3medium_based_on_gpt2](https://huggingface.co/sberbank-ai/rugpt3medium_based_on_gpt2).
- **Датасет:** Датасет новостей содержит столбцы `url`, `title`, `text`, `topic`, `tags` и `date`. В процессе предобработки столбец `title` переименовывается в `headline` для удобства.
- **Преобразование данных:** Из каждой новости формируется промпт вида:
Новость: <текст новости> Заголовок:

При этом токены промпта замаскированы (устанавливаются в -100), чтобы loss рассчитывался только по генерируемой части (заголовку).
- **Методы генерации:** Для демонстрации работы дообученной модели протестированы различные методы генерации:
- Greedy Search
- Beam Search
- Сэмплирование с температурой
- Nucleus (Top-p) Sampling
- **Оптимизации:** Для ускорения обучения используются:
- Режим fp16 (смешанная точность)
- Накопление градиентов (gradient accumulation)
- Увеличенное число DataLoader workers
- Возможное использование `torch.compile` (при наличии PyTorch 2.0+)

## Установка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/RondyC/news-headline-generator/model.py
   cd your-repo-name
Датасет скачивается автоматически с Google Drive. Код использует URL:
https://drive.google.com/uc?id=1QLkXmuK-ul_4eAYKI-fRs7mXk1nIgtsL

После скачивания данные предобрабатываются:
* Столбец title переименовывается в headline.
* Датасет ограничивается до 140000 примеров и разбивается на:
 * Train: 100000 примеров
 * Validation: 20000 примеров
 * Test: 20000 примеров

Дообучение модели
В скрипте дообучения используется Hugging Face Trainer с следующими основными настройками:

* Batch size: 4 (на устройство)
* Gradient accumulation: 32 шагов, что даёт effective batch size 128
* fp16: включён, чтобы ускорить вычисления на GPU
* Дataloader workers: 4 или больше для ускорения загрузки данных

Результаты
* Greedy Search: «...» – заголовок получился информативным.
* Beam Search: «...» – может содержать повторения.
* Сэмплирование с температурой: «...» – разнообразный, но иногда фрагментарный.
* Nucleus Sampling: «...» – наиболее сбалансированный вариант.

Выводы
* Дообученная модель успешно научилась генерировать заголовки, соответствующие содержанию новостей.
* Выбор метода генерации влияет на качество и разнообразие результата: nucleus sampling показал наилучшие результаты для этой задачи.
* Оптимизации (fp16, gradient accumulation, увеличение числа DataLoader workers и torch.compile) позволили ускорить обучение, хотя можно экспериментировать с дальнейшими оптимизациями для повышения скорости.


Благодарности
* Hugging Face Transformers
* Datasets
* Lenta.ru News Dataset
* Sberbank AI для предоставления предобученной модели.
