<div align="center">

# Machine Learning Portfolio

### От данных до проверки результата

Шесть учебных проектов Антона Чуркина: поиск, временные ряды и компьютерное зрение.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-2563EB?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
[![Checks](https://github.com/Xokage9/ml-portfolio/actions/workflows/checks.yml/badge.svg)](https://github.com/Xokage9/ml-portfolio/actions/workflows/checks.yml)

[Профиль](https://github.com/Xokage9) · [Telegram](https://t.me/xokage9) · [Как отобраны работы](docs/selection.md)

</div>

## Проекты

| Проект | Основная сложность | Что можно посмотреть |
|---|---|---|
| [RAG по научным статьям](projects/01_rag) | Чанкинг, FAISS, генерация и проверка источников | 350 статей, 2 стратегии, 5 сценариев |
| [Заправки по телеметрии](projects/02_fuel_events) | BiLSTM, абляции, событийные метрики | Event F1 и point F1 на общих временных точках |
| [Детекция номеров SVHN](projects/03_object_detection) | HDF5-разметка, YOLOv8n, геометрия рамок | Подготовка данных, mAP, IoU и пропуски |
| [Сегментация знаков](projects/04_segmentation) | DeepLabV3, маски, AMP и накопление градиентов | Foreground IoU и визуальные примеры |
| [Цвет автомобилей](projects/05_transfer_learning) | Несбалансированные классы и fine-tuning | ResNet50, ConvNeXt-Tiny и CNN с нуля |
| [Графовый поиск](projects/06_graph_search) | Термины, сообщества, TF-IDF и графы публикаций | 1081 статья, центральности и 15 сообществ |

В каждой папке находятся описание задачи, ноутбук, зависимости и зафиксированные результаты с ограничениями протокола. Крупные наборы данных и веса загружаются отдельно; источники указаны в описании каждой работы.

## Начать просмотр

- **Backend и LLM:** [RAG](projects/01_rag) — получение корпуса, индексирование, запрос и проверка ответа.
- **Эксперимент и метрики:** [временные ряды](projects/02_fuel_events) — рост событийного F1 и его цена в точности границ событий.
- **Computer vision:** [детекция](projects/03_object_detection) и [сегментация](projects/04_segmentation) — от разметки до оценки.

## Воспроизведение

Python 3.11 или 3.12. Для каждой работы лучше создать отдельное виртуальное окружение и установить её `requirements.txt`. Ссылки на Colab находятся в README проектов. GPU нужен для полного обучения моделей; CI не выполняет обучение.

```bash
git clone https://github.com/Xokage9/ml-portfolio.git
cd ml-portfolio
python -m pip install -r requirements-checks.txt
python scripts/validate_notebooks.py
python -m unittest discover -s tests -v
```

## О результатах

Числа и графики в `results/recorded_*` взяты из сохранённых исходных экспериментов. Выводы ноутбуков очищены, полный повтор GPU-обучения при оформлении репозитория не проводился. В нескольких работах исправлен протокол или код, поэтому старые цифры нельзя считать результатами исправленной версии. [Подробности проверки и изменений](docs/validation.md).

Исходные учебные постановки, данные и модели имеют собственных авторов и условия использования. [Источники и происхождение работ](docs/sources.md).
