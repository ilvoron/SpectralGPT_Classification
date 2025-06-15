import streamlit as st
import torch
import numpy as np
import rasterio
import plotly.express as px
import pandas as pd
import torch.nn.functional as F
import traceback
import os
import glob
from types import SimpleNamespace
import models_vit_tensor
from util.datasets import EuroSat
from util.pos_embed import interpolate_pos_embed

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def create_args_namespace(checkpoint_path, input_size=128, patch_size=8, nb_classes=10):
    args = SimpleNamespace()
    args.model_type = 'tensor'
    args.model = 'vit_base_patch8_128'
    args.input_size = input_size
    args.patch_size = patch_size
    args.nb_classes = nb_classes
    args.drop_path = 0.0
    args.global_pool = True
    args.dataset_type = 'euro_sat'
    args.dropped_bands = [10]
    args.masked_bands = None
    args.finetune = checkpoint_path
    args.eval = True
    return args


def scan_checkpoints_folder(checkpoints_dir="./checkpoints"):
    checkpoint_files = []
    if not os.path.exists(checkpoints_dir):
        st.warning(f"Папка с checkpoint'ами не найдена: {checkpoints_dir}")
        return checkpoint_files
    checkpoint_paths = glob.glob(os.path.join(checkpoints_dir, "*.pth"))

    for checkpoint_path in checkpoint_paths:
        if os.path.isfile(checkpoint_path) and os.access(checkpoint_path, os.R_OK):
            checkpoint_files.append({
                "path": checkpoint_path,
                "filename": os.path.basename(checkpoint_path),
                "display_name": os.path.basename(checkpoint_path)
            })

    checkpoint_files.sort(key=lambda x: x['filename'])
    return checkpoint_files


def load_model_from_checkpoint(checkpoint_path, input_size=128, patch_size=8, nb_classes=10):
    try:
        # Создаем args объект для совместимости
        args = create_args_namespace(checkpoint_path, input_size, patch_size, nb_classes)

        if args.model_type == 'tensor':
            model = models_vit_tensor.__dict__[args.model](
                drop_path_rate=args.drop_path,
                num_classes=args.nb_classes
            )
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")

        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # Удаляем несовместимые ключи
        keys_to_check = ['patch_embed.0.proj.weight', 'patch_embed.1.proj.weight',
                         'patch_embed.2.proj.weight', 'patch_embed.2.proj.bias',
                         'head.weight', 'head.bias']
        for k in keys_to_check:
            if k in checkpoint_model and k in state_dict:
                if checkpoint_model[k].shape != state_dict[k].shape:
                    del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)  # Интерполируем позиционные эмбеддинги
        model.load_state_dict(checkpoint_model, strict=False)  # Загружаем веса
        model.eval()  # Переводим в режим evaluation

        return model, args

    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.text("Traceback:")
        st.text(traceback.format_exc())
        return None, None


def create_eurosat_transform(args, is_train=False):
    try:
        # Получаем полные mean и std для всех 13 каналов
        # Не удаляем dropped_bands здесь - это будет сделано после transform
        mean, std = EuroSat.mean, EuroSat.std

        # Создаем transform с полными mean/std
        transform = EuroSat.build_transform(
            is_train=is_train,
            input_size=args.input_size,
            mean=mean,
            std=std
        )

        return transform

    except Exception as e:
        st.error(f"Ошибка создания transform: {str(e)}")
        return None


def preprocess_single_image(image_array, args):
    try:
        # Преобразуем из (c, h, w) в (h, w, c)
        if len(image_array.shape) == 3:
            image_array = image_array.transpose(1, 2, 0)  # (h, w, c)

        # Конвертируем в float32
        image_array = image_array.astype(np.float32)

        # Применяем masked_bands если есть
        if args.masked_bands is not None:
            image_array[:, :, args.masked_bands] = np.array(EuroSat.mean)[args.masked_bands]

        # Создаем transform с полными mean/std
        transform = create_eurosat_transform(args, is_train=False)
        if transform is None:
            return None

        # Применяем transform к полному изображению (13 каналов)
        img_as_tensor = transform(image_array)  # (c, h, w)

        # Удаляем dropped_bands
        if args.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in args.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        # Добавляем batch dimension
        final_tensor = img_as_tensor.unsqueeze(0)

        return final_tensor

    except Exception as e:
        st.error(f"Ошибка предобработки изображения: {str(e)}")
        st.text("Traceback:")
        st.text(traceback.format_exc())
        return None


def predict_image(model, preprocessed_tensor):
    try:
        with torch.no_grad():
            outputs = model(preprocessed_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()[0]
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        st.text("Traceback:")
        st.text(traceback.format_exc())
        return None


def create_band_combination_image(image_array, bands, enhance=True):
    try:
        rgb_bands = image_array[bands, :, :]
        rgb_image = np.zeros((rgb_bands.shape[1], rgb_bands.shape[2], 3))

        for i in range(3):
            band = rgb_bands[i]
            if enhance:
                # Улучшение контраста (обрезаем выбросы)
                p2, p98 = np.percentile(band, (2, 98))
                band = np.clip(band, p2, p98)

            # Нормализация к [0, 1]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                rgb_image[:, :, i] = (band - band_min) / (band_max - band_min)
            else:
                rgb_image[:, :, i] = 0

        return rgb_image
    except Exception as e:
        st.error(f"Ошибка создания композиции: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="EuroSAT SpectralGPT Analyzer", layout="wide")

    st.title("EuroSAT SpectralGPT Analyzer")
    st.markdown("Анализ спектральных изображений EuroSAT с помощью finetune модели SpectralGPT")

    st.sidebar.header("Настройки модели")

    checkpoints_dir = st.sidebar.text_input(
        "Папка с checkpoint'ами:",
        value="./checkpoints",
        help="Укажите путь к папке, содержащей .pth файлы"
    )

    if st.sidebar.button("Обновить список checkpoint'ов"):
        st.rerun()

    checkpoint_files = scan_checkpoints_folder(checkpoints_dir)

    if not checkpoint_files:
        st.sidebar.warning("Checkpoint'ы не найдены!")
        st.sidebar.markdown("Проверьте путь к папке или добавьте .pth файлы")
        selected_checkpoint_path = None
    else:
        st.sidebar.success(f"Найдено checkpoint'ов: {len(checkpoint_files)}")

        checkpoint_options = [info['filename'] for info in checkpoint_files]
        selected_filename = st.sidebar.selectbox(
            "Выберите checkpoint:",
            checkpoint_options,
            help="Выберите .pth файл для загрузки модели"
        )

        selected_checkpoint_path = None
        for checkpoint_info in checkpoint_files:
            if checkpoint_info['filename'] == selected_filename:
                selected_checkpoint_path = checkpoint_info['path']
                break

    col_upload, col_analyze = st.columns([4, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Загрузите TIF файл EuroSAT (13 каналов):",
            type=['tif', 'tiff'],
            help="Выберите многоспектральное изображение в формате TIFF"
        )

    with col_analyze:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("АНАЛИЗИРОВАТЬ", use_container_width=True, type="primary")

    if uploaded_file is not None:
        st.success(f"Файл загружен: {uploaded_file.name}")

        try:
            # Сохраняем временный файл
            if not os.path.exists("temp"):
                os.makedirs("temp")
            with open(f"temp/temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with rasterio.open(f"temp/temp_{uploaded_file.name}") as src:
                image_array = src.read()  # (channels, height, width)

            st.info(
                f"Размер изображения: {image_array.shape} (каналы: {image_array.shape[0]}, высота: {image_array.shape[1]}, ширина: {image_array.shape[2]})")

            if analyze_button:
                if not selected_checkpoint_path:
                    st.error("Выберите checkpoint для анализа!")
                    return

                with st.spinner("Загрузка модели и анализ изображения..."):
                    try:
                        model, args = load_model_from_checkpoint(selected_checkpoint_path)
                        if model is None or args is None:
                            return

                        preprocessed = preprocess_single_image(image_array, args)
                        if preprocessed is None:
                            return

                        probabilities = predict_image(model, preprocessed)
                        if probabilities is None:
                            return

                        results = [(EUROSAT_CLASSES[i], prob * 100) for i, prob in enumerate(probabilities)]
                        results.sort(key=lambda x: x[1], reverse=True)

                        col_rgb, col_results, col_diagram = st.columns(3)

                        with col_rgb:
                            st.subheader("RGB изображение")
                            try:
                                rgb_image = create_band_combination_image(
                                    image_array, [3, 2, 1]  # Red, Green, Blue
                                )
                                if rgb_image is not None:
                                    st.image(
                                        rgb_image,
                                        caption="Естественный цвет (B4-B3-B2)",
                                        use_container_width=True
                                    )
                                else:
                                    st.error("Ошибка создания RGB изображения")
                            except Exception as e:
                                st.error(f"Ошибка в RGB: {str(e)}")

                        with col_results:
                            st.subheader("Результаты классификации")
                            for i, (class_name, probability) in enumerate(results, 1):
                                st.write(f"{i}. **{class_name}**: {probability:.2f}%")

                        with col_diagram:
                            st.subheader("Диаграмма вероятностей")
                            df = pd.DataFrame({
                                'Класс': [result[0] for result in results],
                                'Вероятность (%)': [result[1] for result in results]
                            })
                            fig = px.bar(
                                df,
                                x='Вероятность (%)',
                                y='Класс',
                                orientation='h',
                                title="Распределение вероятностей по классам",
                                color='Вероятность (%)',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(
                                height=400,
                                yaxis={'categoryorder': 'total ascending'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Общая ошибка анализа: {str(e)}")
                        st.text("Полный traceback:")
                        st.text(traceback.format_exc())

            try:
                os.remove(f"temp/temp_{uploaded_file.name}")
            except:
                pass

        except Exception as e:
            st.error(f"Ошибка загрузки файла: {str(e)}")
            st.text("Traceback:")
            st.text(traceback.format_exc())

    else:
        st.info("Загрузите TIF файл для начала анализа")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Информация о классах EuroSAT")
    st.sidebar.markdown("""
    **EuroSAT Dataset** содержит 10 классов землепользования, полученных со спутников Sentinel-2:

    - **AnnualCrop** - Однолетние культуры (пшеница, кукуруза)
    - **Forest** - Лесные массивы (хвойные и лиственные)
    - **HerbaceousVegetation** - Травянистая растительность (луга, степи)
    - **Highway** - Автомобильные дороги и транспортная инфраструктура
    - **Industrial** - Промышленные зоны и производственные объекты
    - **Pasture** - Пастбища для выпаса скота
    - **PermanentCrop** - Многолетние культуры (виноградники, сады)
    - **Residential** - Жилые районы и городская застройка
    - **River** - Реки и речные системы
    - **SeaLake** - Морские акватории и озера

    **Спектральные каналы Sentinel-2** используют электромагнитное излучение в диапазоне от видимого света до коротковолнового инфракрасного. Каждый канал несет уникальную информацию о поверхности Земли, что позволяет различать материалы и объекты, неразличимые для человеческого глаза.

    **Канал B10 (Cirrus)** автоматически исключается из анализа в соответствии с настройками обучения модели SpectralGPT.
    """)


if __name__ == "__main__":
    main()
