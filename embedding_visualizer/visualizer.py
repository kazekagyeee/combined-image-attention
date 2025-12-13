import json
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Tuple
import glob
import textwrap
from sklearn.impute import SimpleImputer


class ProcrustesAnalyzer:
    """
    Класс для выравнивания пространств эмбеддингов с помощью Procrustes Analysis.
    """

    def __init__(self, reference_model: str = "BERT"):
        """
        Args:
            reference_model: Модель, к пространству которой будем приводить другие модели
        """
        self.reference_model = reference_model
        self.transformation_matrices = {}
        self.scalers = {}
        self.disparity_scores = {}

    def analyze_models(self, metadata: List[Dict]) -> Dict:
        """
        Анализирует типы моделей и группирует эмбеддинги.

        Returns:
            Словарь с группированными эмбеддингами по типам моделей
        """
        model_groups = {}

        for i, meta in enumerate(metadata):
            model_type = meta.get('model_type', 'Unknown')
            if model_type not in model_groups:
                model_groups[model_type] = {
                    'indices': [],
                    'dim': meta.get('original_dim', 0),
                    'count': 0
                }
            model_groups[model_type]['indices'].append(i)
            model_groups[model_type]['count'] += 1

        return model_groups

    def prepare_alignment_data(self, embeddings: List[np.ndarray],
                               metadata: List[Dict]) -> Dict:
        """
        Подготавливает данные для выравнивания.

        Returns:
            Словарь с группированными эмбеддингами и метаданными
        """
        model_groups = self.analyze_models(metadata)

        # Преобразуем список эмбеддингов в словарь по моделям
        aligned_data = {}
        for model_type, group in model_groups.items():
            indices = group['indices']
            model_embeddings = [embeddings[i] for i in indices]

            # Приводим к массиву
            if model_embeddings:
                aligned_data[model_type] = {
                    'embeddings': np.vstack(model_embeddings),
                    'indices': indices,
                    'dim': group['dim'],
                    'count': group['count']
                }

        return aligned_data

    def orthogonal_procrustes_analysis(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Orthogonal Procrustes Analysis - находит оптимальное вращение и масштабирование.

        Args:
            X: Исходные эмбеддинги (n_samples, n_features)
            Y: Целевые эмбеддинги (n_samples, n_features)

        Returns:
            aligned_X: Выровненные эмбеддинги X
            disparity: Мера различия (чем меньше, тем лучше)
        """
        # Центрируем данные
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)

        # Вычисляем матрицу корреляции
        C = X_centered.T @ Y_centered

        # SVD разложение
        U, _, Vt = svd(C, full_matrices=False)

        # Оптимальное вращение
        R = U @ Vt

        # Масштабирование
        scale = np.trace(C @ R) / np.trace(X_centered.T @ X_centered)

        # Применяем преобразование
        aligned_X = X_centered @ (scale * R) + np.mean(Y, axis=0)

        # Вычисляем disparity (меру различия)
        disparity = np.sum((aligned_X - Y_centered - np.mean(Y, axis=0)) ** 2)

        return aligned_X, disparity, R, scale

    def generalized_procrustes_analysis(self, embeddings_dict: Dict,
                                        max_iterations: int = 100) -> Dict:
        """
        Generalized Procrustes Analysis - выравнивает несколько наборов одновременно.

        Args:
            embeddings_dict: Словарь {model_type: embeddings_array}
            max_iterations: Максимальное число итераций

        Returns:
            Словарь с выровненными эмбеддингами
        """
        print(f"Обобщенный Procrustes Analysis (до {max_iterations} итераций)...")

        # Инициализируем эталон как среднее всех наборов
        model_keys = list(embeddings_dict.keys())
        n_models = len(model_keys)

        # Проверяем размерности
        shapes = [emb.shape for emb in embeddings_dict.values()]
        if len(set(shape[1] for shape in shapes)) > 1:
            print("Предупреждение: разные размерности эмбеддингов")
            # Приводим к минимальной размерности
            min_dim = min(shape[1] for shape in shapes)
            for key in model_keys:
                if embeddings_dict[key].shape[1] > min_dim:
                    pca = PCA(n_components=min_dim, random_state=42)
                    embeddings_dict[key] = pca.fit_transform(embeddings_dict[key])

        # Инициализируем эталон
        reference = np.mean([embeddings_dict[key] for key in model_keys], axis=0)

        # Итеративное выравнивание
        for iteration in range(max_iterations):
            total_disparity = 0
            aligned_embeddings = {}

            for key in model_keys:
                if key == self.reference_model:
                    aligned_embeddings[key] = embeddings_dict[key]
                else:
                    aligned, disparity, R, scale = self.orthogonal_procrustes_analysis(
                        embeddings_dict[key], reference
                    )
                    aligned_embeddings[key] = aligned
                    total_disparity += disparity

                    # Сохраняем матрицу преобразования
                    self.transformation_matrices[key] = R
                    self.disparity_scores[key] = disparity

            # Обновляем эталон
            new_reference = np.mean([aligned_embeddings[key] for key in model_keys], axis=0)

            # Проверяем сходимость
            if np.allclose(reference, new_reference, rtol=1e-6):
                print(f"Сходимость достигнута на итерации {iteration + 1}")
                break

            reference = new_reference

        return aligned_embeddings

    def align_embeddings(self, embeddings: List[np.ndarray],
                         metadata: List[Dict],
                         alignment_method: str = 'orthogonal') -> np.ndarray:
        """
        Основной метод для выравнивания эмбеддингов.

        Args:
            embeddings: Список эмбеддингов
            metadata: Метаданные для каждого эмбеддинга
            alignment_method: Метод выравнивания ('orthogonal', 'generalized')

        Returns:
            Выровненные эмбеддинги в едином пространстве
        """
        print("\n" + "=" * 60)
        print("PROCRUSTES ANALYSIS - ВЫРАВНИВАНИЕ ПРОСТРАНСТВ")
        print("=" * 60)

        # Подготавливаем данные
        model_data = self.prepare_alignment_data(embeddings, metadata)

        # Проверяем наличие референсной модели
        if self.reference_model not in model_data:
            available_models = list(model_data.keys())
            print(f"Референсная модель '{self.reference_model}' не найдена.")
            print(f"Доступные модели: {available_models}")
            print(f"Использую '{available_models[0]}' как референсную.")
            self.reference_model = available_models[0]

        print(f"Референсная модель: {self.reference_model}")
        print(f"Обнаружено моделей: {len(model_data)}")

        for model_type, data in model_data.items():
            print(f"  {model_type}: {data['count']} объектов, {data['dim']}D")

        # Если только одна модель, выравнивание не требуется
        if len(model_data) == 1:
            print("Только одна модель обнаружена. Выравнивание не требуется.")
            return np.vstack([model_data[model_type]['embeddings']
                              for model_type in model_data])

        # Применяем выбранный метод выравнивания
        if alignment_method == 'orthogonal':
            aligned_embeddings = self._orthogonal_alignment(model_data)
        elif alignment_method == 'generalized':
            aligned_embeddings = self._generalized_alignment(model_data)
        else:
            raise ValueError(f"Неизвестный метод выравнивания: {alignment_method}")

        # Объединяем все выровненные эмбеддинги
        all_aligned = []
        all_indices = []

        for model_type in model_data:
            indices = model_data[model_type]['indices']
            if model_type in aligned_embeddings:
                all_aligned.append(aligned_embeddings[model_type])
            else:
                all_aligned.append(model_data[model_type]['embeddings'])
            all_indices.extend(indices)

        # Восстанавливаем правильный порядок
        combined = np.vstack(all_aligned)
        sorted_indices = np.argsort(all_indices)
        final_embeddings = combined[sorted_indices]

        print(f"\nВыравнивание завершено.")
        print(f"Финальная форма эмбеддингов: {final_embeddings.shape}")

        # Выводим отчет по disparity
        if self.disparity_scores:
            print("\nDisparity scores (чем меньше, тем лучше):")
            for model, score in self.disparity_scores.items():
                print(f"  {model}: {score:.6f}")

        return final_embeddings

    def _orthogonal_alignment(self, model_data: Dict) -> Dict:
        """
        Orthogonal Procrustes выравнивание каждой модели к референсной.
        """
        print("\nПрименение Orthogonal Procrustes...")

        ref_model = self.reference_model
        ref_embeddings = model_data[ref_model]['embeddings']
        aligned_results = {ref_model: ref_embeddings}

        for model_type, data in model_data.items():
            if model_type == ref_model:
                continue

            print(f"  Выравнивание {model_type} → {ref_model}")

            source = data['embeddings']
            target = ref_embeddings

            # Проверяем размерности
            if source.shape[1] != target.shape[1]:
                print(f"    Приведение размерности: {source.shape[1]}D → {target.shape[1]}D")
                if source.shape[1] > target.shape[1]:
                    # Уменьшаем размерность source
                    pca = PCA(n_components=target.shape[1], random_state=42)
                    source = pca.fit_transform(source)
                else:
                    # Дополняем source нулями
                    diff = target.shape[1] - source.shape[1]
                    source = np.pad(source, ((0, 0), (0, diff)), mode='constant')

            # Приводим к одинаковому количеству образцов
            n_samples = min(source.shape[0], target.shape[0])
            source = source[:n_samples]
            target = target[:n_samples]

            # Нормализуем
            source_scaled = StandardScaler().fit_transform(source)
            target_scaled = StandardScaler().fit_transform(target)

            # Применяем Orthogonal Procrustes
            aligned, disparity, R, scale = self.orthogonal_procrustes_analysis(
                source_scaled, target_scaled
            )

            aligned_results[model_type] = aligned
            self.disparity_scores[model_type] = disparity
            self.transformation_matrices[model_type] = R

            print(f"    Disparity: {disparity:.6f}")

        return aligned_results

    def _generalized_alignment(self, model_data: Dict) -> Dict:
        """
        Generalized Procrustes выравнивание всех моделей одновременно.
        """
        print("\nПрименение Generalized Procrustes...")

        # Подготавливаем словарь эмбеддингов
        embeddings_dict = {}
        for model_type, data in model_data.items():
            embeddings_dict[model_type] = data['embeddings']

        # Применяем Generalized Procrustes
        aligned_embeddings = self.generalized_procrustes_analysis(embeddings_dict)

        return aligned_embeddings

    def export_transformations(self, filename: str = "procrustes_transformations.json"):
        """
        Экспортирует матрицы преобразований в JSON.
        """
        transformations = {}

        for model_type, matrix in self.transformation_matrices.items():
            transformations[model_type] = {
                'matrix': matrix.tolist(),
                'shape': list(matrix.shape),
                'disparity': float(self.disparity_scores.get(model_type, 0.0))
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transformations, f, indent=2, ensure_ascii=False)

        print(f"Матрицы преобразований сохранены в: {filename}")


class EnhancedEmbeddingVisualizer:
    """
    Улучшенный класс для 3D визуализации текстовых эмбеддингов.
    Поддерживает Procrustes Analysis для выравнивания пространств разных моделей.
    """

    def __init__(self, intermediate_dim: int = 256, use_procrustes: bool = True):
        """
        Args:
            intermediate_dim: Целевая размерность для промежуточного представления
            use_procrustes: Использовать Procrustes Analysis для выравнивания
        """
        self.intermediate_dim = intermediate_dim
        self.use_procrustes = use_procrustes
        self.procrustes_analyzer = None
        self.embeddings_data = []
        self.metadata = []
        self.df = None
        self.embeddings_processed = None
        self.pca_256d = None
        self.pca_3d = None
        self.scaler = None

        if use_procrustes:
            self.procrustes_analyzer = ProcrustesAnalyzer(reference_model="BERT")

    def load_json_files(self, folder_path: str, file_pattern: str = "*.json"):
        """
        Загружает все JSON файлы из указанной папки.
        """
        json_files = glob.glob(os.path.join(folder_path, file_pattern))

        if not json_files:
            print(f"Файлы не найдены по пути: {folder_path}/{file_pattern}")
            return

        print(f"Найдено {len(json_files)} JSON файлов")

        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)

                    if isinstance(content, list):
                        for item in content:
                            self._process_item(item, os.path.basename(file_path))
                    elif isinstance(content, dict):
                        self._process_item(content, os.path.basename(file_path))

            except Exception as e:
                print(f"Ошибка при загрузке файла {file_path}: {e}")

    def _process_item(self, item: Dict[str, Any], source_file: str):
        """Обрабатывает один элемент JSON."""
        if 'text_embedding' in item:
            embedding = np.array(item['text_embedding'], dtype=np.float32)
            original_dim = len(embedding)

            # Сохраняем эмбеддинг
            self.embeddings_data.append({
                'embedding': embedding,
                'original_dim': original_dim
            })

            # Создаем короткую версию caption для легенды
            caption = item.get('caption', '')
            caption_short = caption[:100] + "..." if len(caption) > 100 else caption

            metadata = {
                'source_file': source_file,
                'caption': caption,
                'caption_short': caption_short,
                'prompt_used': item.get('prompt_used', ''),
                'score': item.get('score', 0.0),
                'area': item.get('area', 0.0),
                'rel_size_coeff': item.get('rel_size_coeff', 0.0),
                'crop_path': item.get('crop_path', ''),
                'orig_path': item.get('orig_path', ''),
                'bbox': str(item.get('bbox', [])),
                'original_dim': original_dim,
                'model_type': self._detect_model_type(embedding, source_file)
            }
            self.metadata.append(metadata)

    def _detect_model_type(self, embedding: np.ndarray, source_file: str) -> str:
        """
        Определяет тип модели по размерности эмбеддинга и имени файла.
        """
        dim = len(embedding)

        if "bert" in source_file.lower():
            return "BERT"
        elif "qwen" in source_file.lower() or "vl" in source_file.lower():
            return "Qwen-VL"
        elif dim == 768:  # BERT-like
            return "BERT-like"
        elif dim == 1024:  # Qwen-like
            return "Qwen-like"
        elif dim == 2048 or dim == 3584:  # Большие модели
            return f"Large_{dim}D"
        else:
            return f"Unknown_{dim}D"

    def prepare_embeddings(self) -> List[np.ndarray]:
        """
        Подготавливает эмбеддинги для обработки.
        """
        if not self.embeddings_data:
            print("Нет загруженных эмбеддингов")
            return []

        # Извлекаем эмбеддинги
        embeddings_list = [item['embedding'] for item in self.embeddings_data]

        print("=" * 60)
        print("АНАЛИЗ ЭМБЕДДИНГОВ")
        print("=" * 60)
        print(f"Количество объектов: {len(self.embeddings_data)}")

        # Анализируем размерности
        unique_dims = set(item['original_dim'] for item in self.embeddings_data)
        model_types = set(m['model_type'] for m in self.metadata)

        print(f"Уникальные размерности: {sorted(unique_dims)}")
        print(f"Типы моделей: {sorted(model_types)}")

        # Статистика по размерностям
        print("\nСтатистика по размерностям:")
        for dim in sorted(unique_dims):
            count = sum(1 for item in self.embeddings_data if item['original_dim'] == dim)
            print(f"  {dim}D: {count} объектов")

        # Статистика по типам моделей
        print("\nСтатистика по типам моделей:")
        for model_type in sorted(model_types):
            count = sum(1 for m in self.metadata if m['model_type'] == model_type)
            dims = set(m['original_dim'] for m in self.metadata if m['model_type'] == model_type)
            print(f"  {model_type}: {count} объектов, размерности: {sorted(dims)}")

        return embeddings_list

    def process_embeddings(self, embeddings_list: List[np.ndarray],
                           alignment_method: str = 'orthogonal') -> np.ndarray:
        """
        Обрабатывает эмбеддинги: выравнивает и приводит к единой размерности.
        """
        if not embeddings_list:
            return np.array([])

        if self.use_procrustes and len(set(m['model_type'] for m in self.metadata)) > 1:
            print("\n" + "=" * 60)
            print("ПРИМЕНЕНИЕ PROCRUSTES ANALYSIS")
            print("=" * 60)

            # Выравниваем пространства с помощью Procrustes
            aligned_embeddings = self.procrustes_analyzer.align_embeddings(
                embeddings_list,
                self.metadata,
                alignment_method=alignment_method
            )

            # Экспортируем матрицы преобразований
            self.procrustes_analyzer.export_transformations()

            # Приводим к единой размерности
            processed_embeddings = self._reduce_to_common_dimension(aligned_embeddings)

        else:
            print("\n" + "=" * 60)
            print("ПРИВЕДЕНИЕ К ЕДИНОЙ РАЗМЕРНОСТИ (БЕЗ PROCRUSTES)")
            print("=" * 60)

            # Просто приводим к единой размерности
            processed_embeddings = self._reduce_to_common_dimension(
                np.vstack(embeddings_list)
            )

        self.embeddings_processed = processed_embeddings
        return processed_embeddings

    def _reduce_to_common_dimension(self, embeddings: np.ndarray) -> np.ndarray:
        print(f"\nПриведение к {self.intermediate_dim}D...")

        if embeddings.shape[0] < 2:
            print(f"Недостаточно данных: {embeddings.shape[0]} образцов")
            return embeddings

        # 1. Импутация NaN значений
        print("Проверка и обработка NaN значений...")
        nan_count_before = np.isnan(embeddings).sum()
        if nan_count_before > 0:
            print(f"   Обнаружено NaN значений: {nan_count_before}")
            imputer = SimpleImputer(strategy='mean')
            embeddings = imputer.fit_transform(embeddings)
            print(f"   NaN значения заменены на среднее по столбцам[citation:1]")
        else:
            print("   NaN значений не обнаружено.")

        # 2. Нормализация данных
        self.scaler = StandardScaler()
        embeddings_scaled = self.scaler.fit_transform(embeddings)

        # Определяем количество компонент для PCA
        n_components = min(self.intermediate_dim, embeddings_scaled.shape[0] - 1,
                           embeddings_scaled.shape[1])

        if n_components < self.intermediate_dim:
            print(f"Внимание: используем {n_components} компонент вместо {self.intermediate_dim}")

        # Применяем PCA
        self.pca_256d = PCA(n_components=n_components, random_state=42)
        embeddings_256d = self.pca_256d.fit_transform(embeddings_scaled)

        explained_variance = self.pca_256d.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        print(f"Объясненная дисперсия PCA (первые 5 компонент): {explained_variance[:5]}")
        print(f"Суммарная объясненная дисперсия: {cumulative_variance[-1]:.4f}")
        print(f"Сохранено дисперсии: {cumulative_variance[-1] * 100:.1f}%")

        print(f"Форма после PCA: {embeddings_256d.shape}")

        return embeddings_256d

    def reduce_to_3d(self, embeddings: np.ndarray, method: str = 'pca') -> np.ndarray:
        """
        Уменьшает размерность для визуализации в 3D.
        """
        print(f"\nПриведение к 3D с помощью {method.upper()}...")

        if embeddings.shape[0] < 3:
            print(f"Недостаточно данных для 3D визуализации")
            return np.array([])

        if method == 'pca':
            n_components = min(3, embeddings.shape[0] - 1)
            self.pca_3d = PCA(n_components=n_components, random_state=42)
            coords_3d = self.pca_3d.fit_transform(embeddings)

            explained_variance = self.pca_3d.explained_variance_ratio_
            print(f"Объясненная дисперсия 3D PCA: {explained_variance}")
            print(f"Суммарная объясненная дисперсия: {sum(explained_variance):.4f}")

        elif method == 'tsne':
            perplexity = min(30, embeddings.shape[0] - 1)
            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=perplexity,
                max_iter=1000,
                learning_rate='auto',
                init='pca'
            )
            coords_3d = tsne.fit_transform(embeddings)
            print(f"Применен t-SNE (perplexity={perplexity})")

        elif method == 'umap':
            try:
                import umap
                n_neighbors = min(15, embeddings.shape[0] - 1)
                reducer = umap.UMAP(
                    n_components=3,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=0.1
                )
                coords_3d = reducer.fit_transform(embeddings)
                print(f"Применен UMAP")
            except ImportError:
                print("UMAP не установлен. Использую PCA.")
                n_components = min(3, embeddings.shape[0] - 1)
                self.pca_3d = PCA(n_components=n_components, random_state=42)
                coords_3d = self.pca_3d.fit_transform(embeddings)

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        # Нормализуем для лучшей визуализации
        coords_3d = StandardScaler().fit_transform(coords_3d)

        print(f"Финальная форма 3D координат: {coords_3d.shape}")

        return coords_3d

    def create_enhanced_3d_plot(self, coords_3d: np.ndarray,
                                color_by: str = 'model_type',
                                size_by: str = '1',
                                title: str = "3D визуализация эмбеддингов") -> go.Figure:
        """
        Создает улучшенный интерактивный 3D график.
        """
        if len(self.metadata) != len(coords_3d):
            raise ValueError(
                f"Количество метаданных ({len(self.metadata)}) не совпадает с количеством координат ({len(coords_3d)})")

        # Создаем DataFrame
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
            'index': range(len(coords_3d))
        })

        # Добавляем метаданные
        for key in self.metadata[0].keys():
            df[key] = [m[key] for m in self.metadata]

        # Создаем форматированный текст для hover
        def format_hover_text(row):
            lines = []
            lines.append(f"<b>Точка #{row['index'] + 1}</b>")
            lines.append("─" * 30)
            lines.append(f"<b>Модель:</b> {row['model_type']}")
            lines.append(f"<b>Исходная размерность:</b> {row['original_dim']}D")
            lines.append(f"<b>Файл:</b> {row['source_file']}")

            if row['caption']:
                lines.append("<br><b>Описание:</b>")
                caption_lines = textwrap.wrap(row['caption'], width=60)
                for line in caption_lines:
                    lines.append(f"  {line}")

            lines.append(f"<br><b>Оценка:</b> {row['score']:.3f}")

            return "<br>".join(lines)

        df['hover_text'] = df.apply(format_hover_text, axis=1)

        # Создаем фигуру
        fig = go.Figure()

        if color_by in df.columns:
            if df[color_by].dtype == 'object':
                unique_values = df[color_by].unique()

                if len(unique_values) <= 10:
                    colors = px.colors.qualitative.Set3
                else:
                    colors = px.colors.qualitative.Light24

                for i, value in enumerate(sorted(unique_values)):
                    mask = df[color_by] == value
                    subset = df[mask]

                    if size_by in df.columns and df[size_by].dtype != 'object':
                        sizes = np.clip(subset[size_by] * 20, 5, 30)
                    else:
                        sizes = 8

                    fig.add_trace(go.Scatter3d(
                        x=subset['x'],
                        y=subset['y'],
                        z=subset['z'],
                        mode='markers',
                        name=str(value),
                        marker=dict(
                            size=sizes,
                            color=colors[i % len(colors)],
                            opacity=0.8,
                            line=dict(width=1, color='white'),
                            symbol='circle'
                        ),
                        text=subset['hover_text'],
                        hoverinfo='text',
                        hovertemplate='%{text}<extra></extra>'
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=df['x'],
                    y=df['y'],
                    z=df['z'],
                    mode='markers',
                    marker=dict(
                        size=df[size_by] * 20 if size_by in df.columns else 8,
                        color=df[color_by],
                        colorscale='Viridis',
                        colorbar=dict(title=color_by),
                        opacity=0.8,
                        line=dict(width=1, color='white'),
                        showscale=True
                    ),
                    text=df['hover_text'],
                    hoverinfo='text',
                    hovertemplate='%{text}<extra></extra>'
                ))

        # Добавляем информацию о методе выравнивания в заголовок
        if self.use_procrustes:
            alignment_info = " (с Procrustes выравниванием)"
        else:
            alignment_info = " (без Procrustes)"

        # Настраиваем макет
        fig.update_layout(
            title=dict(
                text=f"{title}{alignment_info}",
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Компонента X',
                yaxis_title='Компонента Y',
                zaxis_title='Компонента Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        self.df = df
        return fig

    def export_to_html(self, fig: go.Figure, filename: str = "embeddings_3d.html"):
        """
        Экспортирует график в HTML.
        """
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': [
                'drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape'
            ],
            'displaylogo': False
        }

        fig.write_html(
            filename,
            config=config,
            include_plotlyjs='cdn',
            full_html=True,
            auto_open=True
        )
        print(f"График сохранен как: {filename}")

    def export_analysis_report(self, filename: str = "embedding_analysis.txt"):
        """
        Экспортирует отчет по анализу эмбеддингов.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ ПО АНАЛИЗУ ЭМБЕДДИНГОВ\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Всего объектов: {len(self.embeddings_data)}\n")
            f.write(f"Использован Procrustes: {'Да' if self.use_procrustes else 'Нет'}\n")

            if self.use_procrustes and self.procrustes_analyzer:
                f.write(f"Референсная модель: {self.procrustes_analyzer.reference_model}\n")

            # Статистика по моделям
            model_stats = {}
            for m in self.metadata:
                model_type = m['model_type']
                if model_type not in model_stats:
                    model_stats[model_type] = {'count': 0, 'dims': set()}
                model_stats[model_type]['count'] += 1
                model_stats[model_type]['dims'].add(m['original_dim'])

            f.write("\nСтатистика по моделям:\n")
            f.write("-" * 40 + "\n")
            for model_type, stats in model_stats.items():
                f.write(f"{model_type}: {stats['count']} объектов, размерности: {sorted(stats['dims'])}\n")

            if self.pca_256d and hasattr(self.pca_256d, 'explained_variance_ratio_'):
                f.write("\nАнализ PCA (256D):\n")
                f.write("-" * 40 + "\n")
                variance = self.pca_256d.explained_variance_ratio_
                f.write(f"Объясненная дисперсия: {sum(variance):.4f}\n")

            if self.pca_3d and hasattr(self.pca_3d, 'explained_variance_ratio_'):
                f.write("\nАнализ PCA (3D):\n")
                f.write("-" * 40 + "\n")
                variance_3d = self.pca_3d.explained_variance_ratio_
                f.write(f"Объясненная дисперсия в 3D: {sum(variance_3d):.4f}\n")

            if self.use_procrustes and self.procrustes_analyzer and self.procrustes_analyzer.disparity_scores:
                f.write("\nРезультаты Procrustes Analysis:\n")
                f.write("-" * 40 + "\n")
                for model, disparity in self.procrustes_analyzer.disparity_scores.items():
                    f.write(f"{model}: disparity = {disparity:.6f}\n")

        print(f"Отчет сохранен в: {filename}")


def main():
    """
    Основная функция для запуска визуализации.
    """
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ ЭМБЕДДИНГОВ С PROCRUSTES ANALYSIS")
    print("=" * 70)

    # Инициализируем визуализатор
    # use_procrustes=True для включения выравнивания пространств
    visualizer = EnhancedEmbeddingVisualizer(
        intermediate_dim=256,
        use_procrustes=False  # Включить/выключить Procrustes
    )

    # Загружаем данные
    folder_path = "./embeddings"
    visualizer.load_json_files(folder_path)

    if not visualizer.embeddings_data:
        print("Нет данных для визуализации")
        return

    # Подготавливаем эмбеддинги
    embeddings_list = visualizer.prepare_embeddings()

    if not embeddings_list:
        print("Не удалось подготовить эмбеддинги")
        return

    # Обрабатываем эмбеддинги (выравнивание и приведение к 256D)
    processed_embeddings = visualizer.process_embeddings(
        embeddings_list,
        alignment_method='orthogonal'  # 'orthogonal' или 'generalized'
    )

    if processed_embeddings.size == 0:
        print("Не удалось обработать эмбеддинги")
        return

    # Приводим к 3D для визуализации
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ В 3D")
    print("=" * 60)

    method_3d = 'pca'  # 'pca', 'tsne', 'umap'
    coords_3d = visualizer.reduce_to_3d(processed_embeddings, method=method_3d)

    if coords_3d.size == 0:
        print("Не удалось создать 3D координаты")
        return

    # Создаем визуализацию
    print("\nСоздание 3D визуализации...")

    fig = visualizer.create_enhanced_3d_plot(
        coords_3d,
        color_by='model_type',
        size_by='1',
        title=f"3D визуализация эмбеддингов ({method_3d.upper()})"
    )

    # Экспортируем
    if visualizer.use_procrustes:
        filename = "embeddings_procrustes_3d.html"
    else:
        filename = "embeddings_3d.html"

    visualizer.export_to_html(fig, filename)
    visualizer.export_analysis_report("embedding_analysis.txt")

    print("\n" + "=" * 70)
    print("ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"1. График сохранен как: {filename}")
    print(f"2. Отчет по анализу: embedding_analysis.txt")

    if visualizer.use_procrustes:
        print(f"3. Матрицы преобразований: procrustes_transformations.json")
        print("\nВсе эмбеддинги выровнены с помощью Procrustes Analysis")
    else:
        print("\nЭмбеддинги визуализированы без выравнивания пространств")


if __name__ == "__main__":
    main()