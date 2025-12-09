import json
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Any
import glob
import textwrap


class EnhancedEmbedding3DVisualizer:
    """
    Улучшенный класс для 3D визуализации текстовых эмбеддингов.
    """

    def __init__(self):
        self.data = []
        self.embeddings = []
        self.metadata = []
        self.df = None

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
            self.embeddings.append(embedding)

            # Создаем короткую версию caption для легенды
            caption_short = item.get('caption', '')[:100] + "..." if len(item.get('caption', '')) > 100 else item.get(
                'caption', '')

            metadata = {
                'source_file': source_file,
                'caption': item.get('caption', ''),
                'caption_short': caption_short,
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
        """Подготавливает эмбеддинги для визуализации."""
        if not self.embeddings:
            print("Нет загруженных эмбеддингов")
            return np.array([])

        embeddings_array = np.vstack(self.embeddings)
        print(f"Форма эмбеддингов: {embeddings_array.shape}")
        print(f"Количество объектов: {len(self.embeddings)}")

        return embeddings_array

    def reduce_to_3d(self, embeddings: np.ndarray, method: str = 'pca') -> np.ndarray:
        """Уменьшает размерность эмбеддингов до 3D."""
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        if method == 'pca':
            pca = PCA(n_components=3, random_state=42)
            coords_3d = pca.fit_transform(embeddings_scaled)

            explained_variance = pca.explained_variance_ratio_
            print(f"Объясненная дисперсия PCA: {explained_variance}")
            print(f"Суммарная объясненная дисперсия: {sum(explained_variance):.4f}")

        elif method == 'tsne':
            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=min(30, len(embeddings) - 1),
                max_iter=1000,
                learning_rate='auto'
            )
            coords_3d = tsne.fit_transform(embeddings_scaled)

        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=3, random_state=42)
                coords_3d = reducer.fit_transform(embeddings_scaled)
            except ImportError:
                print("UMAP не установлен. Устанавливаю PCA по умолчанию.")
                pca = PCA(n_components=3, random_state=42)
                coords_3d = pca.fit_transform(embeddings_scaled)

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return coords_3d

    def create_enhanced_3d_plot(self, coords_3d: np.ndarray,
                                color_by: str = 'source_file',
                                size_by: str = 'score',
                                title: str = "3D визуализация текстовых эмбеддингов") -> go.Figure:
        """
        Создает улучшенный интерактивный 3D график.

        Особенности:
        - Полный caption при наведении с переносами строк
        - Улучшенная навигация (прокрутка для масштабирования)
        - Плавное вращение
        - Кнопки сброса и сохранения
        """
        if len(self.metadata) != len(coords_3d):
            raise ValueError("Количество метаданных не совпадает с количеством координат")

        # Создаем DataFrame с улучшенными данными
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
            'index': range(len(coords_3d))
        })

        # Добавляем метаданные с форматированием для отображения
        for key in self.metadata[0].keys():
            df[key] = [m[key] for m in self.metadata]

        # Создаем форматированный текст для hover
        def format_hover_text(row):
            """Форматирует текст для отображения при наведении."""
            lines = []

            # Добавляем номер точки
            lines.append(f"<b>Точка #{row['index'] + 1}</b>")
            lines.append("─" * 30)

            # Полный caption с переносами
            caption_lines = textwrap.wrap(row['caption'], width=60)
            if caption_lines:
                lines.append("<b>Описание:</b>")
                for line in caption_lines:
                    lines.append(f"  {line}")
                lines.append("")

            # Метаданные
            lines.append(f"<b>Файл:</b> {row['source_file']}")
            lines.append(f"<b>Оценка:</b> {row['score']:.3f}")
            lines.append(f"<b>Площадь:</b> {row['area']:.1f}")

            # Обрезанный prompt для экономии места
            if row['prompt_used']:
                prompt_short = row['prompt_used'][:80] + "..." if len(row['prompt_used']) > 80 else row['prompt_used']
                lines.append(f"<b>Промпт:</b> {prompt_short}")

            return "<br>".join(lines)

        df['hover_text'] = df.apply(format_hover_text, axis=1)

        # Создаем фигуру с помощью plotly.graph_objects для большего контроля
        fig = go.Figure()

        # Определяем цвета для категорий
        if color_by in df.columns:
            if df[color_by].dtype == 'object':  # Категориальные данные
                unique_values = df[color_by].unique()
                colors = px.colors.qualitative.Set3

                for i, value in enumerate(unique_values):
                    mask = df[color_by] == value
                    subset = df[mask]

                    # Определяем размер точек
                    sizes = subset[size_by] * 20 if size_by in df.columns else 8

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
                            line=dict(width=1, color='white')
                        ),
                        text=subset['hover_text'],
                        hoverinfo='text',
                        hovertemplate='%{text}<extra></extra>',
                        customdata=subset['index']
                    ))
            else:  # Числовые данные
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
                    hovertemplate='%{text}<extra></extra>',
                    customdata=df['index']
                ))

        # Настраиваем макет с улучшенной навигацией
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Компонента 1',
                yaxis_title='Компонента 2',
                zaxis_title='Компонента 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
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
            ),
            # Включаем прокрутку для масштабирования
            scene_camera=dict(
                projection=dict(type='perspective')
            ),
            # Добавляем кнопки навигации
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            label="Сбросить вид",
                            method="relayout",
                            args=[{"scene.camera": dict(eye=dict(x=1.5, y=1.5, z=1.5),
                                                        up=dict(x=0, y=0, z=1),
                                                        center=dict(x=0, y=0, z=0))}]
                        ),
                        dict(
                            label="Вид сверху",
                            method="relayout",
                            args=[{"scene.camera": dict(eye=dict(x=0, y=0, z=2.5),
                                                        up=dict(x=0, y=1, z=0),
                                                        center=dict(x=0, y=0, z=0))}]
                        ),
                        dict(
                            label="Вид сбоку",
                            method="relayout",
                            args=[{"scene.camera": dict(eye=dict(x=2.5, y=0, z=0),
                                                        up=dict(x=0, y=0, z=1),
                                                        center=dict(x=0, y=0, z=0))}]
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )

        # Добавляем аннотации для подсказок
        fig.add_annotation(
            text="<b>Навигация:</b><br>• ЛКМ + движение = вращение<br>• ПКМ + движение = перемещение<br>• Прокрутка = масштаб",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="left"
        )

        self.df = df  # Сохраняем для возможного экспорта
        return fig

    def create_interactive_dashboard(self, coords_3d: np.ndarray):
        """
        Создает интерактивную панель с несколькими видами.
        """
        if len(self.metadata) != len(coords_3d):
            raise ValueError("Количество метаданных не совпадает с количеством координат")

        # Создаем DataFrame
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
            'index': range(len(coords_3d))
        })

        for key in self.metadata[0].keys():
            df[key] = [m[key] for m in self.metadata]

        # Создаем подграфики: 3D вид и таблица с деталями
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d', 'rowspan': 2}, {'type': 'scatter'}],
                   [None, {'type': 'table'}]],
            column_widths=[0.7, 0.3],
            row_heights=[0.7, 0.3],
            subplot_titles=('3D Визуализация эмбеддингов', '2D Проекция', 'Детали точки')
        )

        # 3D график
        fig.add_trace(
            go.Scatter3d(
                x=df['x'],
                y=df['y'],
                z=df['z'],
                mode='markers',
                marker=dict(
                    size=df['score'] * 15,
                    color=df['score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score", x=1.05)
                ),
                text=df['caption_short'],
                hoverinfo='text',
                name='Эмбеддинги'
            ),
            row=1, col=1
        )

        # 2D проекция (xy)
        fig.add_trace(
            go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['score'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=df['caption_short'],
                hoverinfo='text',
                name='2D проекция'
            ),
            row=1, col=2
        )

        # Таблица с деталями (пустая, будет заполняться при клике)
        fig.add_trace(
            go.Table(
                header=dict(values=['Параметр', 'Значение']),
                cells=dict(values=[[''], ['']]),
                name='Детали'
            ),
            row=2, col=2
        )

        # Настраиваем макет
        fig.update_layout(
            title="Интерактивная панель визуализации эмбеддингов",
            height=800,
            showlegend=False,
            hovermode='closest'
        )

        return fig

    def export_to_html(self, fig: go.Figure, filename: str = "embeddings_3d_enhanced.html"):
        """
        Экспортирует график в HTML с дополнительными функциями.
        """
        # Добавляем кастомный CSS для лучшего отображения
        config = {
            'scrollZoom': True,  # Включаем прокрутку для масштабирования
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

        # Сохраняем с дополнительными опциями
        fig.write_html(
            filename,
            config=config,
            include_plotlyjs='cdn',
            full_html=True,
            auto_open=True
        )
        print(f"График сохранен как: {filename}")

    def export_selected_data(self, indices: List[int], filename: str = "selected_points.json"):
        """Экспортирует выбранные точки в JSON."""
        if self.df is None or len(indices) == 0:
            print("Нет данных для экспорта")
            return

        selected_data = []
        for idx in indices:
            if idx < len(self.df):
                row = self.df.iloc[idx]
                point_data = {
                    'index': int(row['index']),
                    'coordinates': {
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'z': float(row['z'])
                    },
                    'caption': row['caption'],
                    'source_file': row['source_file'],
                    'score': float(row['score']),
                    'prompt_used': row['prompt_used']
                }
                selected_data.append(point_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(selected_data, f, indent=2, ensure_ascii=False)

        print(f"Экспортировано {len(selected_data)} точек в {filename}")


def enhanced_main():
    """
    Улучшенная основная функция для демонстрации.
    """
    visualizer = EnhancedEmbedding3DVisualizer()

    # Загружаем все JSON файлы из папки
    folder_path = "./embeddings"
    visualizer.load_json_files(folder_path)

    # Подготавливаем эмбеддинги
    embeddings = visualizer.prepare_embeddings()

    if embeddings.size == 0:
        print("Нет данных для визуализации")
        return

    # Уменьшаем размерность до 3D с помощью PCA (быстрее и стабильнее для начала)
    print("\nУменьшение размерности с помощью PCA...")
    coords_3d_pca = visualizer.reduce_to_3d(embeddings, method='pca')

    # Создаем улучшенную визуализацию
    print("\nСоздание улучшенной 3D визуализации...")

    # Вариант 1: Основной улучшенный график
    fig1 = visualizer.create_enhanced_3d_plot(
        coords_3d_pca,
        color_by='source_file',
        size_by='score',
        title="Улучшенная 3D визуализация текстовых эмбеддингов"
    )

    # Экспортируем с улучшенными функциями
    visualizer.export_to_html(fig1, "embeddings_3d_enhanced.html")

    print("\n=== Инструкция по использованию ===")
    print("1. Откройте файл embeddings_3d_enhanced.html в браузере")
    print("2. Наведите курсор на точку - увидите полное описание")
    print("3. Используйте мышь для навигации:")
    print("   - ЛКМ + движение: вращение")
    print("   - ПКМ + движение: перемещение")
    print("   - Прокрутка: масштабирование")
    print("4. Используйте кнопки сверху для смены вида")

    # Дополнительно: создаем дашборд с несколькими видами
    print("\nСоздание интерактивной панели...")
    dashboard = visualizer.create_interactive_dashboard(coords_3d_pca)
    visualizer.export_to_html(dashboard, "embeddings_dashboard.html")

    # Показываем статистику
    print("\n=== Статистика данных ===")
    print(f"Всего объектов: {len(visualizer.embeddings)}")
    print(f"Размерность эмбеддингов: {embeddings.shape[1]}")

    if visualizer.metadata:
        unique_files = set(m['source_file'] for m in visualizer.metadata)
        print(f"Уникальные файлы: {len(unique_files)}")
        print(f"Средняя длина caption: {np.mean([len(m['caption']) for m in visualizer.metadata]):.0f} символов")
        print(
            f"Диапазон оценок: {min([m['score'] for m in visualizer.metadata]):.2f} - {max([m['score'] for m in visualizer.metadata]):.2f}")


def quick_visualize(folder_path: str = "./embeddings"):
    """
    Быстрая визуализация с минимумом настроек.
    """
    visualizer = EnhancedEmbedding3DVisualizer()
    visualizer.load_json_files(folder_path)

    embeddings = visualizer.prepare_embeddings()
    if embeddings.size == 0:
        return None

    coords_3d = visualizer.reduce_to_3d(embeddings, method='pca')

    fig = visualizer.create_enhanced_3d_plot(
        coords_3d,
        color_by='score',
        size_by='area',
        title="Быстрая 3D визуализация эмбеддингов"
    )

    visualizer.export_to_html(fig, "quick_visualization.html")
    return fig


if __name__ == "__main__":
    # Запускаем улучшенную визуализацию
    enhanced_main()

    # Для быстрого тестирования можно использовать:
    # quick_visualize("./embeddings")