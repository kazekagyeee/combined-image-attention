import json
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import glob


class Embedding3DVisualizer:
    """
    Класс для создания 3D визуализации текстовых эмбеддингов из JSON файлов.
    """

    def __init__(self):
        self.data = []
        self.embeddings = []
        self.metadata = []

    def load_json_files(self, folder_path: str, file_pattern: str = "*.json"):
        """
        Загружает все JSON файлы из указанной папки.

        Args:
            folder_path: Путь к папке с JSON файлами
            file_pattern: Шаблон для поиска файлов (по умолчанию *.json)
        """
        # Находим все JSON файлы в папке
        json_files = glob.glob(os.path.join(folder_path, file_pattern))

        if not json_files:
            print(f"Файлы не найдены по пути: {folder_path}/{file_pattern}")
            return

        print(f"Найдено {len(json_files)} JSON файлов")

        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)

                    # Проверяем, является ли content списком объектов
                    if isinstance(content, list):
                        for item in content:
                            self._process_item(item, os.path.basename(file_path))
                    elif isinstance(content, dict):
                        self._process_item(content, os.path.basename(file_path))

            except Exception as e:
                print(f"Ошибка при загрузке файла {file_path}: {e}")

    def _process_item(self, item: Dict[str, Any], source_file: str):
        """
        Обрабатывает один элемент JSON.
        """
        if 'text_embedding' in item:
            # Извлекаем эмбеддинг
            embedding = np.array(item['text_embedding'], dtype=np.float32)

            # Сохраняем эмбеддинг
            self.embeddings.append(embedding)

            # Сохраняем метаданные
            metadata = {
                'source_file': source_file,
                'caption': item.get('caption', ''),
                'prompt_used': item.get('prompt_used', ''),
                'score': item.get('score', 0.0),
                'area': item.get('area', 0.0),
                'rel_size_coeff': item.get('rel_size_coeff', 0.0),
                'crop_path': item.get('crop_path', ''),
                'orig_path': item.get('orig_path', ''),
                'bbox': str(item.get('bbox', []))
            }
            self.metadata.append(metadata)

    def prepare_embeddings(self) -> np.ndarray:
        """
        Подготавливает эмбеддинги для визуализации.

        Returns:
            Массив эмбеддингов
        """
        if not self.embeddings:
            print("Нет загруженных эмбеддингов")
            return np.array([])

        # Преобразуем в numpy массив
        embeddings_array = np.vstack(self.embeddings)
        print(f"Форма эмбеддингов: {embeddings_array.shape}")
        print(f"Количество объектов: {len(self.embeddings)}")

        return embeddings_array

    def reduce_to_3d(self, embeddings: np.ndarray, method: str = 'pca') -> np.ndarray:
        """
        Уменьшает размерность эмбеддингов до 3D.

        Args:
            embeddings: Массив эмбеддингов
            method: Метод уменьшения размерности ('pca', 'tsne', 'umap')

        Returns:
            3D координаты
        """
        # Масштабируем данные
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        if method == 'pca':
            # Используем PCA для уменьшения до 3D
            pca = PCA(n_components=3, random_state=42)
            coords_3d = pca.fit_transform(embeddings_scaled)

            # Выводим информацию о объясненной дисперсии
            explained_variance = pca.explained_variance_ratio_
            print(f"Объясненная дисперсия PCA: {explained_variance}")
            print(f"Суммарная объясненная дисперсия: {sum(explained_variance):.4f}")

        elif method == 'tsne':
            # Используем t-SNE для уменьшения до 3D
            tsne = TSNE(n_components=3, random_state=42,
                        perplexity=min(30, len(embeddings) - 1),
                        max_iter=1000)
            coords_3d = tsne.fit_transform(embeddings_scaled)

        elif method == 'umap':
            try:
                import umap
                # Используем UMAP для уменьшения до 3D
                reducer = umap.UMAP(n_components=3, random_state=42)
                coords_3d = reducer.fit_transform(embeddings_scaled)
            except ImportError:
                print("UMAP не установлен. Устанавливаю PCA по умолчанию.")
                pca = PCA(n_components=3, random_state=42)
                coords_3d = pca.fit_transform(embeddings_scaled)

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return coords_3d

    def create_3d_plot(self, coords_3d: np.ndarray,
                       color_by: str = 'source_file',
                       size_by: str = 'score',
                       title: str = "3D визуализация текстовых эмбеддингов") -> go.Figure:
        """
        Создает интерактивный 3D график.

        Args:
            coords_3d: 3D координаты точек
            color_by: Поле для окрашивания точек
            size_by: Поле для определения размера точек
            title: Заголовок графика

        Returns:
            Plotly Figure объект
        """
        if len(self.metadata) != len(coords_3d):
            raise ValueError("Количество метаданных не совпадает с количеством координат")

        # Создаем DataFrame для удобства
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
        })

        # Добавляем метаданные
        for key in self.metadata[0].keys():
            df[key] = [m[key] for m in self.metadata]

        # Создаем интерактивную 3D визуализацию
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color=color_by,
            size=size_by if size_by in df.columns else None,
            hover_data=['caption', 'source_file', 'score', 'prompt_used'],
            title=title,
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        # Настраиваем макет
        fig.update_layout(
            scene=dict(
                xaxis_title='Компонента 1',
                yaxis_title='Компонента 2',
                zaxis_title='Компонента 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=True
        )

        return fig

    def create_animated_3d_plot(self, coords_3d: np.ndarray,
                                animation_by: str = 'source_file') -> go.Figure:
        """
        Создает анимированную 3D визуализацию.

        Args:
            coords_3d: 3D координаты точек
            animation_by: Поле для анимации

        Returns:
            Plotly Figure объект
        """
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
        })

        # Добавляем метаданные
        for key in self.metadata[0].keys():
            df[key] = [m[key] for m in self.metadata]

        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color=animation_by,
            animation_frame=animation_by,
            hover_data=['caption', 'score'],
            title="Анимированная 3D визуализация эмбеддингов",
            opacity=0.8
        )

        return fig

    def save_plot(self, fig: go.Figure, filename: str = "embeddings_3d_plot.html"):
        """
        Сохраняет график в HTML файл.

        Args:
            fig: Plotly Figure объект
            filename: Имя файла для сохранения
        """
        fig.write_html(filename)
        print(f"График сохранен как: {filename}")

    def create_static_matplotlib_plot(self, coords_3d: np.ndarray):
        """
        Создает статическую 3D визуализацию с помощью matplotlib.
        Полезно для сохранения в PNG/PDF.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Получаем уникальные источники для цветов
            sources = list(set(m['source_file'] for m in self.metadata))
            colors = plt.cm.tab20(np.linspace(0, 1, len(sources)))

            # Создаем словарь цветов
            color_dict = {source: colors[i] for i, source in enumerate(sources)}

            # Рисуем точки
            for i, (coord, metadata) in enumerate(zip(coords_3d, self.metadata)):
                ax.scatter(
                    coord[0], coord[1], coord[2],
                    c=[color_dict[metadata['source_file']]],
                    s=metadata.get('score', 1) * 50,  # Размер зависит от score
                    alpha=0.7,
                    edgecolors='w',
                    linewidth=0.5
                )

            ax.set_xlabel('Компонента 1')
            ax.set_ylabel('Компонента 2')
            ax.set_zlabel('Компонента 3')
            ax.set_title('3D визуализация текстовых эмбеддингов')

            # Добавляем легенду
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color_dict[src], label=src)
                               for src in sources]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig('embeddings_3d_static.png', dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("Matplotlib не установлен. Используйте plotly для визуализации.")


def main():
    """
    Основная функция для демонстрации работы визуализатора.
    """
    # Создаем визуализатор
    visualizer = Embedding3DVisualizer()

    # Загружаем все JSON файлы из папки
    # Замените путь на ваш реальный путь к папке
    folder_path = "./embeddings"
    visualizer.load_json_files(folder_path)

    # Подготавливаем эмбеддинги
    embeddings = visualizer.prepare_embeddings()

    if embeddings.size == 0:
        print("Нет данных для визуализации")
        return

    # Уменьшаем размерность до 3D
    print("\nУменьшение размерности с помощью PCA...")
    coords_3d_pca = visualizer.reduce_to_3d(embeddings, method='pca')

    print("\nУменьшение размерности с помощью t-SNE...")
    coords_3d_tsne = visualizer.reduce_to_3d(embeddings, method='tsne')

    # Создаем визуализацию с разными параметрами
    print("\nСоздание 3D визуализаций...")

    # Вариант 1: Цвет по исходному файлу, размер по score
    fig1 = visualizer.create_3d_plot(
        coords_3d_pca,
        color_by='source_file',
        size_by='score',
        title="3D визуализация эмбеддингов (PCA) - цвет по файлу"
    )
    visualizer.save_plot(fig1, "embeddings_by_source.html")

    # Вариант 2: Цвет по score (дискретизированный)
    # Добавляем категориальную колонку для score
    for metadata in visualizer.metadata:
        score = metadata['score']
        if score > 0.8:
            metadata['score_category'] = 'Высокий'
        elif score > 0.5:
            metadata['score_category'] = 'Средний'
        else:
            metadata['score_category'] = 'Низкий'

    fig2 = visualizer.create_3d_plot(
        coords_3d_tsne,
        color_by='score_category',
        size_by='area',
        title="3D визуализация эмбеддингов (t-SNE) - цвет по оценке"
    )
    visualizer.save_plot(fig2, "embeddings_by_score.html")

    # Вариант 3: Статическая визуализация с matplotlib
    print("\nСоздание статической визуализации...")
    visualizer.create_static_matplotlib_plot(coords_3d_pca)

    # Показать информацию о данных
    print("\n=== Сводка данных ===")
    print(f"Всего объектов: {len(visualizer.embeddings)}")
    print(f"Размерность эмбеддингов: {embeddings.shape[1]}")
    print(f"Уникальные файлы: {len(set(m['source_file'] for m in visualizer.metadata))}")

    # Выводим примеры данных для проверки
    print("\nПримеры captions:")
    for i, metadata in enumerate(visualizer.metadata[:3]):
        caption_preview = metadata['caption'][:100] + "..." if len(metadata['caption']) > 100 else metadata['caption']
        print(f"{i + 1}. {caption_preview}")
        print(f"   Файл: {metadata['source_file']}, Score: {metadata['score']}")
        print()


if __name__ == "__main__":
    main()