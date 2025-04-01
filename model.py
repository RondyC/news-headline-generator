# Установка необходимых библиотек
"""

!pip install transformers datasets

"""# Используемые библиотеки"""

import os
import gdown
import torch
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

"""# Скачивание датасета с Google Drive (CSV-файл)"""

# URL файла: https://drive.google.com/file/d/1QLkXmuK-ul_4eAYKI-fRs7mXk1nIgtsL/view?usp=sharing
# Из него извлекаем file_id и формируем прямую ссылку:
url = "https://drive.google.com/uc?id=1QLkXmuK-ul_4eAYKI-fRs7mXk1nIgtsL"
output_csv = "russian_news_dataset.csv"
if not os.path.exists(output_csv):
    gdown.download(url, output_csv, quiet=False)

"""# Загрузка датасета из CSV-файла."""

dataset = load_dataset("csv", data_files={"data": output_csv})["data"]

# Предполагается, что в CSV есть столбцы "title" (заголовок новости) и "text" (текст новости).
# Переименуем столбец "title" в "headline" для удобства.
def rename_columns(example):
    example["headline"] = example["title"]
    return example

dataset = dataset.map(rename_columns)

"""# Разделение датасета на train (80%), validation (10%) и test (10%)."""

# Ограничиваем общий датасет до 140000 примеров
total_samples = 140000
dataset = dataset.shuffle(seed=42).select(range(total_samples))

# Разбиваем ограниченный датасет:
train_dataset = dataset.select(range(100000))
val_dataset = dataset.select(range(100000, 120000))
test_dataset = dataset.select(range(120000, total_samples))

# Создаем объект DatasetDict для удобного доступа к сплитам
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

"""# Загрузка модели и токенизатора."""

DEVICE = torch.device("cuda:0")
# Используем русскоязычную модель GPT от Сбера.
model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Функция для формирования промпта и маскирования токенов промпта (для вычисления лосса только по заголовку).
def preprocess_function(examples):
    input_ids_list = []
    labels_list = []
    for text, headline in zip(examples['text'], examples['headline']):
        prompt = f"Новость: {text}\nЗаголовок: "
        full_text = prompt + headline
        tokenized = tokenizer(full_text, truncation=True, max_length=512)
        input_ids = tokenized["input_ids"]
        prompt_ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
        prompt_length = len(prompt_ids)
        labels = input_ids.copy()
        labels[:prompt_length] = [-100] * prompt_length
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Data collator для динамического дополнения последовательностей в батче.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

"""# Fine-tuning: Настройка аргументов и обучение модели"""

def custom_data_collator(features):
    # Извлекаем списки input_ids и labels из каждого примера
    input_ids_list = [feature["input_ids"] for feature in features]
    labels_list = [feature["labels"] for feature in features]

    # Паддим input_ids, используя стандартный метод токенизатора
    batch_inputs = tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt")["input_ids"]
    # Паддим labels так же
    batch_labels = tokenizer.pad({"input_ids": labels_list}, return_tensors="pt")["input_ids"]

    # Заменяем токены паддинга (pad_token_id) в labels на -100
    batch_labels[batch_labels == tokenizer.pad_token_id] = -100

    return {"input_ids": batch_inputs, "labels": batch_labels}

training_args = TrainingArguments(
    output_dir="./finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=5,
    gradient_accumulation_steps=32,
    fp16=True,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_data_collator,
    train_dataset=tokenized_datasets["train"],
    optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-5), None)
)

# Запуск обучения (fine-tuning)
trainer.train()

# Сохранение дообученной модели и токенизатора
model.save_pretrained("./gpt2_russian_headlines")
tokenizer.save_pretrained("./gpt2_russian_headlines")

"""# Демонстрация дообученной модели"""

import random

def generate_headline_for_method(news_text, method, max_new_tokens=70):
    """
    Генерирует заголовок для заданного текста новости по указанному методу.
    Из результата берется только первая строка после разделителя.
    """
    prompt = f"Новость: {news_text}\nЗаголовок: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    if method == "greedy":
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
    elif method == "beam":
        outputs = model.generate(input_ids, num_beams=5, max_new_tokens=max_new_tokens)
    elif method == "temperature":
        outputs = model.generate(input_ids, do_sample=True, temperature=1.3, max_new_tokens=max_new_tokens)
    elif method == "nucleus":
        outputs = model.generate(input_ids, do_sample=True, top_k=20, top_p=0.8, max_new_tokens=max_new_tokens)
    else:
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Заголовок: " in generated:
        # Берем часть после разделителя и оставляем только первую строку
        headline = generated.split("Заголовок: ", 1)[1].strip().split("\n")[0].strip()
    else:
        headline = generated.strip().split("\n")[0].strip()
    return headline


# Выбираем случайную новость из тестовой выборки
import random
sample = random.choice(dataset["test"])
news_text = sample["text"]

methods = ["greedy", "beam", "temperature", "nucleus"]

print("\nДемонстрация генерации заголовков разными методами:\n")
for method in methods:
    headline = generate_headline_for_method(news_text, method)
    print(f"Метод {method}: {headline}\n")
print("Исходная новость:")
print(news_text)

"""# Итоги


---

1. Выбранный датасет:

Для обучения использовался датасет новостей с сайта Lenta.ru, скачанный с Google Drive. Он содержит следующие ключевые столбцы:
* url: ссылка на новость;
* title: заголовок новости (переименован в headline для удобства);
* text: полный текст новости;
* topic, tags, date: дополнительные метаданные (не используются при дообучении).
После предобработки датасет был ограничен до 140000 примеров и разделён на тренировочную (100000 примеров), валидационную (20000 примеров) и тестовую (20000 примеров) выборки. Такой подход позволяет эффективно работать с большим объёмом данных, сохраняя репрезентативность каждого сплита.

2. Дообучение модели:

Для решения задачи генерации заголовков была дообучена русскоязычная модель sberbank-ai/rugpt3medium_based_on_gpt2.
* Подготовка входа: Модель обучалась на промптах вида:
 * Новость: <текст новости>
 * Заголовок:

При этом часть токенов, соответствующая промпту, замаскирована (устанавливается значение –100), чтобы loss рассчитывался только для генерируемой части (заголовка).
* Оптимизации: Использовались современные методы оптимизации – режим fp16 для ускорения вычислений и уменьшения использования памяти, а также накопление градиентов (gradient accumulation), что позволяет эмулировать больший размер батча при ограничениях GPU.

3. Методы генерации заголовков:

Для демонстрации работы дообученной модели были протестированы следующие методы генерации:

* Greedy Search:

При детерминированном выборе токенов модель генерирует стабильный и информативный заголовок, хотя разнообразие остаётся ограниченным.

* Beam Search:

Использование нескольких лучей (num_beams=5) позволяет рассмотреть несколько вариантов, однако метод может приводить к повторениям и избыточности в генерации.

* Сэмплирование с температурой (Temperature Sampling):

Добавление вероятностного сэмплирования (temperature=1.3) даёт больше свободы в выборе токенов, что приводит к более разнообразным, но иногда фрагментарным заголовкам.

* Nucleus Sampling (Top-p Sampling):

Ограничение выбора токенов набором наиболее вероятных вариантов (top_k=20, top_p=0.8) обеспечивает баланс между точностью и разнообразием, что позволило получить наиболее сбалансированные и информативные заголовки.

4. Заключение:

* Качество обучения: Дообучение модели показало стабильное снижение training loss, что свидетельствует о том, что модель успешно усвоила структуру входных данных и научилась генерировать заголовки, соответствующие содержанию новостей.

* Выбор метода генерации: Результаты демонстрации указывают, что для решения задачи генерации заголовков оптимальным может оказаться nucleus sampling, обеспечивающий баланс между разнообразием и точностью, хотя в зависимости от задачи можно использовать и другие методы.
"""