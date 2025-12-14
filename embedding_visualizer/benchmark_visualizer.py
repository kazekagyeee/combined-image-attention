import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from scipy.spatial.distance import cosine, euclidean
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from config import BenchmarkConfig

warnings.filterwarnings('ignore')



class EmbeddingBenchmark:
    def __init__(self, config=BenchmarkConfig):
        self.embeddings = {}
        self.labels = []
        self.sources = []
        self.parent_index = None
        self.core_embedding_path = config.core_embedding_path
        self.embeddings_under_research = config.embeddings_under_research
        self.base_embeddings_path = config.base_embeddings_path

    def load_embeddings(self):
        """Загрузка эмбеддингов из всех файлов"""

        # 1. Загрузка родительского эмбеддинга (core_embedding.json)
        with open(self.core_embedding_path, 'r', encoding='utf-8') as f:
            core_data = json.load(f)
        self.embeddings['parent'] = np.array(core_data[0]['text_embedding']).reshape(1, -1)
        self.labels.append('Родитель')
        self.sources.append('parent')
        self.parent_index = 0

        print(f"Родительский эмбеддинг: {self.embeddings['parent'].shape}")

        # 2. Загрузка исследуемых эмбеддингов (uied-qwen-2.5-2-images.json)
        with open(self.embeddings_under_research, 'r', encoding='utf-8') as f:
            uied_data = json.load(f)

        uied_embeddings = []
        print(f"количество исследуемых {len(uied_data)}")
        for i, item in enumerate(uied_data):
            if 'text_embedding' in item:
                embedding = np.array(item['text_embedding'])
                uied_embeddings.append(embedding)
                self.labels.append(f'Исследуемый {i + 1}')
                self.sources.append('test')

        self.embeddings['test'] = np.array(uied_embeddings)
        print(f"Исследуемые эмбеддинги: {self.embeddings['test'].shape}")

        # 3. Загрузка базовых эмбеддингов (base_model_embeddings.json)
        try:
            with open(self.base_embeddings_path, 'r', encoding='utf-8') as f:
                base_data = json.load(f)

            base_embeddings = []
            for i, item in enumerate(base_data):
                if 'text_embedding' in item:
                    # Проверяем, является ли это полным эмбеддингом
                    embedding = np.array(item['text_embedding'])
                    if len(embedding) > 100:  # Проверка на валидность
                        base_embeddings.append(embedding)
                        self.labels.append(f'Базовый {i + 1}')
                        self.sources.append('base')

            self.embeddings['base'] = np.array(base_embeddings)
            print(f"Базовые эмбеддинги: {self.embeddings['base'].shape}")
        except Exception as e:
            print(f"Ошибка загрузки базовых эмбеддингов: {e}")
            self.embeddings['base'] = np.array([])

    def align_dimensions(self):
        """Выравнивание размерностей эмбеддингов"""
        print("\nВыравнивание размерностей...")

        # Получаем размерности всех эмбеддингов
        parent_dim = self.embeddings['parent'].shape[1]
        test_dim = self.embeddings['test'].shape[1]

        # Инициализируем базовую размерность, если есть базовые эмбеддинги
        base_dim = self.embeddings['base'].shape[1] if len(self.embeddings['base']) > 0 else float('inf')

        # Находим минимальную размерность среди всех эмбеддингов
        min_dim = min(parent_dim, test_dim, base_dim)

        print(f"Размерности: Родитель={parent_dim}, Исследуемые={test_dim}, Базовые={base_dim}")
        print(f"Минимальная размерность: {min_dim}")

        # Функция для обрезки или дополнения эмбеддингов
        def adjust_embedding(embedding, target_dim):
            current_dim = embedding.shape[1]
            if current_dim > target_dim:
                # Обрезаем до target_dim (берем первые target_dim компонентов)
                print(f"  Обрезка с {current_dim} до {target_dim}")
                return embedding[:, :target_dim]
            elif current_dim < target_dim:
                # Дополняем нулями
                print(f"  Дополнение с {current_dim} до {target_dim}")
                padding = np.zeros((embedding.shape[0], target_dim - current_dim))
                return np.hstack([embedding, padding])
            return embedding

        # Выравниваем все эмбеддинги
        print("Выравнивание родительского эмбеддинга...")
        self.embeddings['parent'] = adjust_embedding(self.embeddings['parent'], min_dim)

        print("Выравнивание исследуемых эмбеддингов...")
        self.embeddings['test'] = adjust_embedding(self.embeddings['test'], min_dim)

        if len(self.embeddings['base']) > 0:
            print("Выравнивание базовых эмбеддингов...")
            self.embeddings['base'] = adjust_embedding(self.embeddings['base'], min_dim)

        print("Размерности выровнены успешно")

        # Проверяем результат
        print(f"\nРезультирующие размерности:")
        print(f"  Родитель: {self.embeddings['parent'].shape}")
        print(f"  Исследуемые: {self.embeddings['test'].shape}")
        if len(self.embeddings['base']) > 0:
            print(f"  Базовые: {self.embeddings['base'].shape}")

    def combine_all_embeddings(self):
        """Объединение всех эмбеддингов в одну матрицу"""
        all_embeddings = [self.embeddings['parent'][0]]

        for emb in self.embeddings['test']:
            all_embeddings.append(emb)

        if len(self.embeddings['base']) > 0:
            for emb in self.embeddings['base']:
                all_embeddings.append(emb)

        return np.array(all_embeddings)

    def calculate_distances(self, embeddings_matrix):
        """Вычисление расстояний от каждого эмбеддинга до родительского"""
        parent_embedding = embeddings_matrix[self.parent_index]
        distances = []

        for i, embedding in enumerate(embeddings_matrix):
            if i != self.parent_index:  # Пропускаем родительский
                cos_dist = cosine(parent_embedding, embedding)
                euc_dist = euclidean(parent_embedding, embedding)
                distances.append({
                    'index': i,
                    'label': self.labels[i],
                    'source': self.sources[i],
                    'cosine_distance': cos_dist,
                    'euclidean_distance': euc_dist
                })

        return distances

    def calculate_metrics(self, distances):
        """Вычисление всех метрик"""
        # Разделяем расстояния по источникам
        test_distances = [d for d in distances if d['source'] == 'test']
        base_distances = [d for d in distances if d['source'] == 'base']

        all_cosine_distances = [d['cosine_distance'] for d in distances]

        # Порог T - среднее расстояние
        threshold_T = np.mean(all_cosine_distances)

        metrics = {
            'threshold_T': threshold_T,
            'test_metrics': {},
            'base_metrics': {},
            'comparison': {}
        }

        # Метрики для исследуемых эмбеддингов
        if test_distances:
            test_cosine = [d['cosine_distance'] for d in test_distances]
            metrics['test_metrics'] = {
                'binary_detection_rate': sum(1 for d in test_cosine if d > threshold_T) / len(test_cosine) * 100,
                'mean_cosine_distance': np.mean(test_cosine),
                'std_cosine_distance': np.std(test_cosine),
                'min_cosine_distance': np.min(test_cosine),
                'max_cosine_distance': np.max(test_cosine),
                'count': len(test_distances)
            }

        # Метрики для базовых эмбеддингов
        if base_distances:
            base_cosine = [d['cosine_distance'] for d in base_distances]
            metrics['base_metrics'] = {
                'binary_detection_rate': sum(1 for d in base_cosine if d > threshold_T) / len(base_cosine) * 100,
                'mean_cosine_distance': np.mean(base_cosine),
                'std_cosine_distance': np.std(base_cosine),
                'min_cosine_distance': np.min(base_cosine),
                'max_cosine_distance': np.max(base_cosine),
                'count': len(base_distances)
            }

        # Сравнительные метрики
        if test_distances and base_distances:
            metrics['comparison'] = {
                'test_better_than_base': metrics['test_metrics']['binary_detection_rate'] > metrics['base_metrics'][
                    'binary_detection_rate'],
                'test_mean_vs_base': metrics['test_metrics']['mean_cosine_distance'] - metrics['base_metrics'][
                    'mean_cosine_distance'],
                'test_std_vs_base': metrics['test_metrics']['std_cosine_distance'] - metrics['base_metrics'][
                    'std_cosine_distance']
            }

        return metrics, test_distances, base_distances

    def perform_pca_analysis(self, embeddings_matrix):
        """Выполнение PCA анализа и создание визуализаций"""
        print("\nВыполнение PCA анализа...")

        # Стандартизация данных
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)

        # PCA для визуализации (3 компоненты)
        pca_3d = PCA(n_components=3)
        pca_result_3d = pca_3d.fit_transform(embeddings_scaled)

        # PCA для анализа дисперсии
        pca_full = PCA()
        pca_full.fit(embeddings_scaled)

        # Объясненная дисперсия
        explained_variance = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        pca_info = {
            'pca_3d_result': pca_result_3d,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'pca_model': pca_3d,
            'components_3d': pca_full.components_[:3]
        }

        print(f"Объясненная дисперсия первыми 3 компонентами: {sum(explained_variance[:3]):.2%}")
        print(f"Объясненная дисперсия по компонентам:")
        for i, var in enumerate(explained_variance[:10], 1):
            print(f"  Компонента {i}: {var:.2%}")

        return pca_info

    def create_3d_visualization(self, pca_result, metrics, distances):
        """Создание 3D визуализации PCA"""

        # Цвета для разных источников
        colors = []
        sizes = []
        hover_texts = []

        for i, source in enumerate(self.sources):
            if source == 'parent':
                colors.append('red')
                sizes.append(20)
                hover_texts.append(f"{self.labels[i]}<br>Родительский элемент")
            elif source == 'test':
                # Находим соответствующее расстояние
                dist_info = next((d for d in distances if d['index'] == i), None)
                if dist_info:
                    colors.append('blue')
                    hover_texts.append(
                        f"{self.labels[i]}<br>"
                        f"Косинусное расстояние: {dist_info['cosine_distance']:.4f}<br>"
                        f"Евклидово расстояние: {dist_info['euclidean_distance']:.4f}"
                    )
                else:
                    colors.append('blue')
                    hover_texts.append(self.labels[i])
                sizes.append(15)
            elif source == 'base':
                colors.append('green')
                dist_info = next((d for d in distances if d['index'] == i), None)
                if dist_info:
                    hover_texts.append(
                        f"{self.labels[i]}<br>"
                        f"Косинусное расстояние: {dist_info['cosine_distance']:.4f}<br>"
                        f"Евклидово расстояние: {dist_info['euclidean_distance']:.4f}"
                    )
                else:
                    hover_texts.append(self.labels[i])
                sizes.append(15)

        # Создаем 3D scatter plot
        fig = go.Figure()

        # Добавляем точки
        fig.add_trace(go.Scatter3d(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            z=pca_result[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.8,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Эмбеддинги'
        ))

        # Настройки layout
        fig.update_layout(
            title='3D PCA Визуализация Эмбеддингов',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            width=1200,
            height=800,
            showlegend=True,
            legend=dict(
                x=0.8,
                y=0.9,
                itemsizing='constant'
            )
        )

        # Добавляем легенду
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Родитель',
            showlegend=True
        ))

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Исследуемые',
            showlegend=True
        ))

        if any(s == 'base' for s in self.sources):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Базовые',
                showlegend=True
            ))

        # Сохраняем HTML файл
        fig.write_html("pca_3d_visualization.html")
        print("3D визуализация сохранена в pca_3d_visualization.html")

        return fig

    def create_variance_plot(self, pca_info):
        """Создание графика объясненной дисперсии"""
        explained_variance = pca_info['explained_variance']
        cumulative_variance = pca_info['cumulative_variance']

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Объясненная дисперсия по компонентам', 'Накопленная объясненная дисперсия'))

        # Первый график: объясненная дисперсия
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(explained_variance[:20]) + 1)),
                y=explained_variance[:20],
                name='Объясненная дисперсия',
                marker_color='blue'
            ),
            row=1, col=1
        )

        # Второй график: накопленная дисперсия
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_variance[:20]) + 1)),
                y=cumulative_variance[:20],
                mode='lines+markers',
                name='Накопленная дисперсия',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )

        # Добавляем горизонтальную линию на 95%
        fig.add_hline(y=0.95, line_dash="dash", line_color="green", row=1, col=2)

        # Настройки layout
        fig.update_xaxes(title_text="Компоненты", row=1, col=1)
        fig.update_xaxes(title_text="Компоненты", row=1, col=2)
        fig.update_yaxes(title_text="Доля дисперсии", row=1, col=1)
        fig.update_yaxes(title_text="Накопленная доля", row=1, col=2)

        fig.update_layout(
            title='Анализ Объясненной Дисперсии PCA',
            height=500,
            width=1200,
            showlegend=True
        )

        fig.write_html("pca_variance_analysis.html")
        print("Анализ дисперсии сохранен в pca_variance_analysis.html")

        return fig

    def create_distance_distribution(self, distances, metrics):
        """Создание графика распределения расстояний"""

        test_distances = [d['cosine_distance'] for d in distances if d['source'] == 'test']
        base_distances = [d['cosine_distance'] for d in distances if d['source'] == 'base']

        fig = go.Figure()

        # Гистограмма для исследуемых эмбеддингов
        if test_distances:
            fig.add_trace(go.Histogram(
                x=test_distances,
                name='Исследуемые',
                opacity=0.7,
                nbinsx=20,
                marker_color='blue'
            ))

        # Гистограмма для базовых эмбеддингов
        if base_distances:
            fig.add_trace(go.Histogram(
                x=base_distances,
                name='Базовые',
                opacity=0.7,
                nbinsx=20,
                marker_color='green'
            ))

        # Добавляем вертикальную линию для порога T
        fig.add_vline(
            x=metrics['threshold_T'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Порог T={metrics['threshold_T']:.4f}",
            annotation_position="top right"
        )

        fig.update_layout(
            title='Распределение Косинусных Расстояний до Родительского Эмбеддинга',
            xaxis_title='Косинусное расстояние',
            yaxis_title='Частота',
            barmode='overlay',
            height=500,
            width=800
        )

        fig.write_html("distance_distribution.html")
        print("Распределение расстояний сохранено в distance_distribution.html")

        return fig

    def create_metrics_report(self, metrics, distances):
        """Создание HTML отчета с метриками"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Бенчмарк Эмбеддингов - Отчет</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 40px; }
                .metrics { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .metric-item { margin: 10px 0; }
                .metric-value { font-weight: bold; color: #2c3e50; }
                .comparison { background: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .distance-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .distance-table th, .distance-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .distance-table th { background-color: #4CAF50; color: white; }
                .test-row { background-color: #e8f4fc; }
                .base-row { background-color: #f0f8f0; }
                .parent-row { background-color: #ffe6e6; }
                .good { color: green; font-weight: bold; }
                .bad { color: red; font-weight: bold; }
                .summary { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Бенчмарк Эмбеддингов - Анализ Качества</h1>
                    <p>Сравнение исследуемых и базовых эмбеддингов</p>
                </div>
        """

        # Основные метрики
        html_content += """
                <div class="metrics">
                    <h2>Основные Метрики</h2>
                    <div class="metric-item">
                        Порог T (среднее расстояние): <span class="metric-value">{:.4f}</span>
                    </div>
        """.format(metrics['threshold_T'])

        if metrics['test_metrics']:
            html_content += """
                    <h3>Исследуемые эмбеддинги ({} шт.)</h3>
                    <div class="metric-item">
                        Binary Detection Rate: <span class="metric-value">{:.2f}%</span>
                    </div>
                    <div class="metric-item">
                        Среднее косинусное расстояние: <span class="metric-value">{:.4f}</span>
                    </div>
                    <div class="metric-item">
                        Стандартное отклонение: <span class="metric-value">{:.4f}</span>
                    </div>
                    <div class="metric-item">
                        Диапазон: [{:.4f}, {:.4f}]
                    </div>
            """.format(
                metrics['test_metrics']['count'],
                metrics['test_metrics']['binary_detection_rate'],
                metrics['test_metrics']['mean_cosine_distance'],
                metrics['test_metrics']['std_cosine_distance'],
                metrics['test_metrics']['min_cosine_distance'],
                metrics['test_metrics']['max_cosine_distance']
            )

        if metrics['base_metrics']:
            html_content += """
                    <h3>Базовые эмбеддинги ({} шт.)</h3>
                    <div class="metric-item">
                        Binary Detection Rate: <span class="metric-value">{:.2f}%</span>
                    </div>
                    <div class="metric-item">
                        Среднее косинусное расстояние: <span class="metric-value">{:.4f}</span>
                    </div>
                    <div class="metric-item">
                        Стандартное отклонение: <span class="metric-value">{:.4f}</span>
                    </div>
                    <div class="metric-item">
                        Диапазон: [{:.4f}, {:.4f}]
                    </div>
            """.format(
                metrics['base_metrics']['count'],
                metrics['base_metrics']['binary_detection_rate'],
                metrics['base_metrics']['mean_cosine_distance'],
                metrics['base_metrics']['std_cosine_distance'],
                metrics['base_metrics']['min_cosine_distance'],
                metrics['base_metrics']['max_cosine_distance']
            )

        html_content += """
                </div>
        """

        # Сравнение
        if metrics['comparison']:
            html_content += """
                <div class="comparison">
                    <h2>Сравнение Метрик</h2>
            """

            if metrics['comparison']['test_better_than_base']:
                html_content += """
                    <div class="metric-item">
                        Исследуемые эмбеддинги <span class="good">лучше</span> базовых по Binary Detection Rate
                    </div>
                """
            else:
                html_content += """
                    <div class="metric-item">
                        Исследуемые эмбеддинги <span class="bad">хуже</span> базовых по Binary Detection Rate
                    </div>
                """

            diff = metrics['comparison']['test_mean_vs_base']
            if diff > 0:
                html_content += """
                    <div class="metric-item">
                        Среднее расстояние исследуемых <span class="good">больше</span> на {:.4f}
                    </div>
                """.format(diff)
            else:
                html_content += """
                    <div class="metric-item">
                        Среднее расстояние исследуемых <span class="bad">меньше</span> на {:.4f}
                    </div>
                """.format(abs(diff))

            html_content += """
                </div>
            """

        # Таблица расстояний
        html_content += """
                <h2>Детальные Расстояния</h2>
                <table class="distance-table">
                    <tr>
                        <th>Тип</th>
                        <th>Метка</th>
                        <th>Косинусное расстояние</th>
                        <th>Евклидово расстояние</th>
                        <th>Превышает порог T</th>
                    </tr>
        """

        for dist in distances:
            row_class = "test-row" if dist['source'] == 'test' else "base-row"
            exceeds_threshold = dist['cosine_distance'] > metrics['threshold_T']
            exceeds_text = '<span class="good">Да</span>' if exceeds_threshold else '<span class="bad">Нет</span>'

            html_content += """
                    <tr class="{}">
                        <td>{}</td>
                        <td>{}</td>
                        <td>{:.4f}</td>
                        <td>{:.4f}</td>
                        <td>{}</td>
                    </tr>
            """.format(
                row_class,
                'Исследуемый' if dist['source'] == 'test' else 'Базовый',
                dist['label'],
                dist['cosine_distance'],
                dist['euclidean_distance'],
                exceeds_text
            )

        html_content += """
                </table>
        """

        # Итоговый вывод
        html_content += """
                <div class="summary">
                    <h2>Итоговый Вывод</h2>
        """

        if metrics.get('test_metrics', {}).get('binary_detection_rate', 0) > 50:
            html_content += """
                    <p>Исследуемые эмбеддинги показывают <span class="good">хорошие результаты</span> 
                    (Binary Detection Rate > 50%). Модель успешно отделяет элементы интерфейса от фона.</p>
            """
        else:
            html_content += """
                    <p>Исследуемые эмбеддинги показывают <span class="bad">неудовлетворительные результаты</span> 
                    (Binary Detection Rate ≤ 50%). Модель плохо отделяет элементы интерфейса от фона.</p>
            """

        if metrics.get('comparison', {}).get('test_better_than_base', False):
            html_content += """
                    <p>Исследуемый метод <span class="good">превосходит</span> базовую модель.</p>
            """
        else:
            html_content += """
                    <p>Исследуемый метод <span class="bad">уступает</span> базовой модели.</p>
            """

        html_content += """
                </div>

                <div style="margin-top: 40px; text-align: center;">
                    <p>Ссылки на визуализации:</p>
                    <ul style="list-style: none; padding: 0;">
                        <li><a href="pca_3d_visualization.html" target="_blank">3D PCA Визуализация</a></li>
                        <li><a href="pca_variance_analysis.html" target="_blank">Анализ дисперсии PCA</a></li>
                        <li><a href="distance_distribution.html" target="_blank">Распределение расстояний</a></li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        with open("benchmark_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print("HTML отчет сохранен в benchmark_report.html")

    def run_benchmark(self):
        """Запуск полного бенчмарк-теста"""
        print("=" * 60)
        print("Запуск бенчмарк-теста эмбеддингов")
        print("=" * 60)

        # Шаг 1: Загрузка эмбеддингов
        self.load_embeddings()

        # Шаг 2: Выравнивание размерностей
        self.align_dimensions()

        # Шаг 3: Объединение всех эмбеддингов
        all_embeddings = self.combine_all_embeddings()
        print(f"\nОбщее количество эмбеддингов: {all_embeddings.shape}")

        # Шаг 4: Вычисление расстояний
        distances = self.calculate_distances(all_embeddings)

        # Шаг 5: Вычисление метрик
        metrics, test_distances, base_distances = self.calculate_metrics(distances)

        # Шаг 6: PCA анализ
        pca_info = self.perform_pca_analysis(all_embeddings)

        # Шаг 7: Создание визуализаций
        self.create_3d_visualization(pca_info['pca_3d_result'], metrics, distances)
        self.create_variance_plot(pca_info)
        self.create_distance_distribution(distances, metrics)

        # Шаг 8: Создание отчета
        self.create_metrics_report(metrics, distances)

        # Вывод результатов в консоль
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ БЕНЧМАРК-ТЕСТА")
        print("=" * 60)

        print(f"\nПорог T (среднее расстояние): {metrics['threshold_T']:.4f}")

        if metrics['test_metrics']:
            print(f"\n--- Исследуемые эмбеддинги ---")
            print(f"Binary Detection Rate: {metrics['test_metrics']['binary_detection_rate']:.2f}%")
            print(f"Среднее расстояние: {metrics['test_metrics']['mean_cosine_distance']:.4f}")
            print(f"Стандартное отклонение: {metrics['test_metrics']['std_cosine_distance']:.4f}")

        if metrics['base_metrics']:
            print(f"\n--- Базовые эмбеддинги ---")
            print(f"Binary Detection Rate: {metrics['base_metrics']['binary_detection_rate']:.2f}%")
            print(f"Среднее расстояние: {metrics['base_metrics']['mean_cosine_distance']:.4f}")
            print(f"Стандартное отклонение: {metrics['base_metrics']['std_cosine_distance']:.4f}")

        if metrics['comparison']:
            print(f"\n--- Сравнение ---")
            if metrics['comparison']['test_better_than_base']:
                print("✓ Исследуемые эмбеддинги лучше базовых")
            else:
                print("✗ Исследуемые эмбеддинги хуже базовых")

        print("\n" + "=" * 60)
        print("Созданы следующие файлы:")
        print("1. pca_3d_visualization.html - 3D визуализация PCA")
        print("2. pca_variance_analysis.html - Анализ дисперсии")
        print("3. distance_distribution.html - Распределение расстояний")
        print("4. benchmark_report.html - Полный отчет с метриками")
        print("=" * 60)


# Запуск бенчмарк-теста
if __name__ == "__main__":
    benchmark = EmbeddingBenchmark()
    benchmark.run_benchmark()