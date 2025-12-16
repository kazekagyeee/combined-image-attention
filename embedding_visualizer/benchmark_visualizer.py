import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        self.threshold_T = getattr(config, 'threshold_T', None)

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

        # Используем порог T из конфига, если задан, иначе вычисляем как среднее
        if self.threshold_T is not None:
            threshold_T = float(self.threshold_T)
            threshold_source = "заданный в конфигурации"
        else:
            threshold_T = np.mean(all_cosine_distances)
            threshold_source = "вычисленный как среднее"

        metrics = {
            'threshold_T': threshold_T,
            'threshold_source': threshold_source,
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

        # Комплексные сравнительные метрики
        if test_distances and base_distances:
            # Получаем значения метрик
            test_bdr = metrics['test_metrics']['binary_detection_rate']
            base_bdr = metrics['base_metrics']['binary_detection_rate']

            test_mean = metrics['test_metrics']['mean_cosine_distance']
            base_mean = metrics['base_metrics']['mean_cosine_distance']

            test_std = metrics['test_metrics']['std_cosine_distance']
            base_std = metrics['base_metrics']['std_cosine_distance']

            # Вычисляем относительные улучшения/ухудшения
            # Для Binary Detection Rate: больше = лучше
            bdr_improvement = test_bdr - base_bdr

            # Для среднего расстояния: БОЛЬШЕ = ЛУЧШЕ (элементы лучше отделяются)
            mean_improvement = test_mean - base_mean

            # Для стандартного отклонения: теперь БОЛЬШЕ тоже = ЛУЧШЕ (до определенного предела)
            # Но с оговоркой - слишком высокое std может быть плохо

            # Нормализуем стандартное отклонение с учетом оптимального диапазона
            # Идеальный std: 0.1-0.3 (умеренная дифференциация)
            optimal_std_min = 0.1
            optimal_std_max = 0.3

            def score_std(std_value):
                if std_value < optimal_std_min:
                    # Слишком низкое - плохо (нет дифференциации)
                    return std_value / optimal_std_min  # < 1.0
                elif std_value <= optimal_std_max:
                    # В оптимальном диапазоне
                    return 1.0
                else:
                    # Слишком высокое - плохо (чрезмерная вариативность)
                    return optimal_std_max / std_value  # < 1.0

            test_std_score = score_std(test_std)
            base_std_score = score_std(base_std)
            std_improvement = test_std_score - base_std_score

            # Взвешенная оценка с новой семантикой
            weights = {
                'bdr': 0.4,  # Binary Detection Rate - важная
                'mean': 0.4,  # Среднее расстояние - столь же важно
                'std': 0.2  # Стандартное отклонение - важно с оговорками
            }

            # Нормализуем улучшения
            max_bdr = max(abs(test_bdr), abs(base_bdr), 1)
            max_mean = max(abs(test_mean), abs(base_mean), 0.01)
            max_std_score = max(abs(test_std_score), abs(base_std_score), 0.01)

            normalized_bdr = bdr_improvement / max_bdr if max_bdr > 0 else 0
            normalized_mean = mean_improvement / max_mean if max_mean > 0 else 0
            normalized_std = std_improvement / max_std_score if max_std_score > 0 else 0

            # Композитный скор
            composite_score = (
                    weights['bdr'] * normalized_bdr +
                    weights['mean'] * normalized_mean +
                    weights['std'] * normalized_std
            )

            metrics['comparison'] = {
                # Базовые сравнения
                'test_better_by_bdr': test_bdr >= base_bdr,
                'test_better_by_mean': test_mean > base_mean,

                # Для std: теперь сложнее - нужно учитывать оптимальный диапазон
                'test_std_in_optimal_range': optimal_std_min <= test_std <= optimal_std_max,
                'base_std_in_optimal_range': optimal_std_min <= base_std <= optimal_std_max,
                'test_better_by_std_score': test_std_score > base_std_score,
                'test_better_by_std': test_std_score > base_std_score,

                # Значения
                'bdr_difference': bdr_improvement,
                'mean_difference': mean_improvement,
                'std_difference': std_improvement,
                'test_std_score': test_std_score,
                'base_std_score': base_std_score,

                # Относительные улучшения (%)
                'bdr_improvement_percent': (bdr_improvement / base_bdr * 100) if base_bdr != 0 else 0,
                'mean_improvement_percent': (mean_improvement / base_mean * 100) if base_mean != 0 else 0,
                'std_improvement_percent': (std_improvement / base_std_score * 100) if base_std_score != 0 else 0,

                # Композитная оценка
                'composite_score': composite_score,
                'test_better_by_composite': composite_score > 0,

                # Правило большинства (адаптированное)
                'test_better_by_majority': (
                                                   (test_bdr > base_bdr) +
                                                   (test_mean > base_mean) +
                                                   (test_std_score > base_std_score)
                                           ) >= 2,

                # Все значения для отчета
                'test_values': {
                    'bdr': test_bdr,
                    'mean': test_mean,
                    'std': test_std,
                    'std_score': test_std_score
                },
                'base_values': {
                    'bdr': base_bdr,
                    'mean': base_mean,
                    'std': base_std,
                    'std_score': base_std_score
                },

                # Интерпретации
                'interpretations': {
                    'mean': {
                        'test': 'Больше = лучше отделение' if test_mean > base_mean else 'Меньше = хуже отделение',
                        'test_value': test_mean,
                        'thresholds': {
                            'excellent': '> 0.3',
                            'good': '0.15-0.3',
                            'poor': '< 0.15'
                        }
                    },
                    'std': {
                        'raw': {
                            'test': test_std,
                            'base': base_std
                        },
                        'score': {
                            'test': test_std_score,
                            'base': base_std_score,
                            'interpretation': '1.0 = оптимально, <1.0 = неоптимально'
                        },
                        'optimal_range': f'{optimal_std_min}-{optimal_std_max}',
                        'interpretation': 'Умеренное std показывает дифференциацию элементов'
                    }
                }
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
        # Используем двойные фигурные скобки для экранирования в CSS
        html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Бенчмарк Эмбеддингов - Отчет</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .metrics {{ background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .metric-item {{ margin: 10px 0; }}
            .metric-value {{ font-weight: bold; color: #2c3e50; }}
            .comparison {{ background: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .comparison-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
            .comparison-table th {{ background-color: #4CAF50; color: white; }}
            .test-row {{ background-color: #e8f4fc; }}
            .base-row {{ background-color: #f0f8f0; }}
            .good {{ color: green; font-weight: bold; }}
            .bad {{ color: red; font-weight: bold; }}
            .neutral {{ color: orange; }}
            .summary {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
            .threshold-info {{ background: #fffacd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .metric-comparison {{ display: flex; justify-content: space-between; margin: 15px 0; }}
            .metric-box {{ flex: 1; padding: 15px; margin: 0 10px; border-radius: 8px; text-align: center; }}
            .test-metric {{ background: #e3f2fd; border: 2px solid #2196f3; }}
            .base-metric {{ background: #e8f5e8; border: 2px solid #4caf50; }}
            .improvement {{ font-size: 0.9em; margin-top: 5px; }}
            .distance-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .distance-table th, .distance-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .distance-table th {{ background-color: #4CAF50; color: white; }}
            .parent-row {{ background-color: #ffe6e6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Бенчмарк Эмбеддингов - Анализ Качества</h1>
                <p>Сравнение исследуемых и базовых эмбеддингов</p>
            </div>

            <div class="threshold-info">
                <h3>Информация о пороге T</h3>
                <div class="metric-item">
                    Значение порога T: <span class="metric-value">{threshold_T:.4f}</span>
                </div>
                <div class="metric-item">
                    Источник порога: <span class="metric-value">{threshold_source}</span>
                </div>
            </div>
    '''.format(
            threshold_T=metrics['threshold_T'],
            threshold_source=metrics['threshold_source']
        )

        # Основные метрики
        html_content += '''
            <div class="metrics">
                <h2>Основные Метрики</h2>
    '''

        if metrics['test_metrics']:
            html_content += '''
                <h3>Исследуемые эмбеддинги ({test_count} шт.)</h3>
                <div class="metric-item">
                    Binary Detection Rate: <span class="metric-value">{test_bdr:.2f}%</span>
                </div>
                <div class="metric-item">
                    Среднее косинусное расстояние: <span class="metric-value">{test_mean:.4f}</span>
                </div>
                <div class="metric-item">
                    Стандартное отклонение: <span class="metric-value">{test_std:.4f}</span>
                </div>
                <div class="metric-item">
                    Диапазон: [{test_min:.4f}, {test_max:.4f}]
                </div>
    '''.format(
                test_count=metrics['test_metrics']['count'],
                test_bdr=metrics['test_metrics']['binary_detection_rate'],
                test_mean=metrics['test_metrics']['mean_cosine_distance'],
                test_std=metrics['test_metrics']['std_cosine_distance'],
                test_min=metrics['test_metrics']['min_cosine_distance'],
                test_max=metrics['test_metrics']['max_cosine_distance']
            )

        if metrics['base_metrics']:
            html_content += '''
                <h3>Базовые эмбеддинги ({base_count} шт.)</h3>
                <div class="metric-item">
                    Binary Detection Rate: <span class="metric-value">{base_bdr:.2f}%</span>
                </div>
                <div class="metric-item">
                    Среднее косинусное расстояние: <span class="metric-value">{base_mean:.4f}</span>
                </div>
                <div class="metric-item">
                    Стандартное отклонение: <span class="metric-value">{base_std:.4f}</span>
                </div>
                <div class="metric-item">
                    Диапазон: [{base_min:.4f}, {base_max:.4f}]
                </div>
    '''.format(
                base_count=metrics['base_metrics']['count'],
                base_bdr=metrics['base_metrics']['binary_detection_rate'],
                base_mean=metrics['base_metrics']['mean_cosine_distance'],
                base_std=metrics['base_metrics']['std_cosine_distance'],
                base_min=metrics['base_metrics']['min_cosine_distance'],
                base_max=metrics['base_metrics']['max_cosine_distance']
            )

        html_content += '''
            </div>
    '''

        # Сравнительные метрики
        if metrics.get('comparison'):
            html_content += '''
            <div class="comparison">
                <h2>Детальное сравнение метрик</h2>

                <div class="metric-comparison">
                    <div class="metric-box test-metric">
                        <h4>Binary Detection Rate</h4>
                        <div class="metric-value">{test_bdr:.2f}%</div>
                        <div class="improvement {bdr_class}">{bdr_sign}{bdr_improvement:.2f}%</div>
                    </div>
                    <div class="metric-box test-metric">
                        <h4>Среднее расстояние</h4>
                        <div class="metric-value">{test_mean:.4f}</div>
                        <div class="improvement {mean_class}">{mean_sign}{mean_improvement:.2f}%</div>
                    </div>
                    <div class="metric-box test-metric">
                        <h4>Стандартное отклонение</h4>
                        <div class="metric-value">{test_std:.4f}</div>
                        <div class="improvement {std_class}">{std_sign}{std_improvement:.2f}%</div>
                    </div>
                </div>

                <div class="metric-comparison">
                    <div class="metric-box base-metric">
                        <h4>Binary Detection Rate</h4>
                        <div class="metric-value">{base_bdr:.2f}%</div>
                        <div class="improvement">Базовое значение</div>
                    </div>
                    <div class="metric-box base-metric">
                        <h4>Среднее расстояние</h4>
                        <div class="metric-value">{base_mean:.4f}</div>
                        <div class="improvement">Базовое значение</div>
                    </div>
                    <div class="metric-box base-metric">
                        <h4>Стандартное отклонение</h4>
                        <div class="metric-value">{base_std:.4f}</div>
                        <div class="improvement">Базовое значение</div>
                    </div>
                </div>

                <table class="comparison-table">
                    <tr>
                        <th>Метрика</th>
                        <th>Исследуемый метод</th>
                        <th>Базовый метод</th>
                        <th>Улучшение</th>
                        <th>Результат</th>
                    </tr>
    '''.format(
                test_bdr=metrics['comparison']['test_values']['bdr'],
                bdr_class='good' if metrics['comparison']['test_better_by_bdr'] else 'bad',
                bdr_sign='+' if metrics['comparison']['bdr_improvement_percent'] >= 0 else '',
                bdr_improvement=metrics['comparison']['bdr_improvement_percent'],
                test_mean=metrics['comparison']['test_values']['mean'],
                mean_class='good' if metrics['comparison']['test_better_by_mean'] else 'bad',
                mean_sign='+' if metrics['comparison']['mean_improvement_percent'] > 0 else '',
                mean_improvement=metrics['comparison']['mean_improvement_percent'],
                test_std=metrics['comparison']['test_values']['std'],
                std_class='good' if metrics['comparison']['test_better_by_std'] else 'bad',
                std_sign='+' if metrics['comparison']['std_improvement_percent'] > 0 else '',
                std_improvement=metrics['comparison']['std_improvement_percent'],
                base_bdr=metrics['comparison']['base_values']['bdr'],
                base_mean=metrics['comparison']['base_values']['mean'],
                base_std=metrics['comparison']['base_values']['std']
            )

            # Добавляем строки таблицы
            comparisons = [
                ('Binary Detection Rate', 'bdr', metrics['comparison']['test_better_by_bdr']),
                ('Среднее расстояние', 'mean', metrics['comparison']['test_better_by_mean']),
                ('Стандартное отклонение', 'std', metrics['comparison']['test_better_by_std'])
            ]

            for name, key, is_better in comparisons:
                test_val = metrics['comparison']['test_values'][key]
                base_val = metrics['comparison']['base_values'][key]
                improvement = metrics['comparison'][f'{key}_improvement_percent']

                html_content += '''
                    <tr>
                        <td>{name}</td>
                        <td>{test_val:.4f}</td>
                        <td>{base_val:.4f}</td>
                        <td class="{improve_class}">{sign}{improvement:.2f}%</td>
                        <td class="{result_class}">{result_text}</td>
                    </tr>
    '''.format(
                    name=name,
                    test_val=test_val,
                    base_val=base_val,
                    improve_class='good' if improvement > 0 else 'bad',
                    sign='+' if improvement > 0 else '',
                    improvement=improvement,
                    result_class='good' if is_better else 'bad',
                    result_text='✓ Лучше' if is_better else '✗ Хуже'
                )

            html_content += '''
                </table>

                <h3>Итоговые оценки</h3>
                <div class="metric-item">
                    Композитная оценка: <span class="metric-value {comp_class}">{composite_score:.4f}</span>
                    <span class="{comp_class2}">{comp_sign}{composite_score_percent:.2%}</span>
                </div>
                <div class="metric-item">
                    Правило большинства (2 из 3): <span class="{majority_class}">{majority_text}</span>
                </div>
                <div class="metric-item">
                    Общий вердикт: <span class="{verdict_class}">{verdict_text}</span>
                </div>
            </div>
    '''.format(
                comp_class='good' if metrics['comparison']['composite_score'] > 0 else 'bad',
                composite_score=metrics['comparison']['composite_score'],
                comp_class2='good' if metrics['comparison']['composite_score'] > 0 else 'bad',
                comp_sign='+' if metrics['comparison']['composite_score'] > 0 else '',
                composite_score_percent=metrics['comparison']['composite_score'],
                majority_class='good' if metrics['comparison']['test_better_by_majority'] else 'bad',
                majority_text='✓ Исследуемый метод лучше' if metrics['comparison'][
                    'test_better_by_majority'] else '✗ Базовый метод лучше',
                verdict_class='good' if metrics['comparison']['test_better_by_composite'] else 'bad',
                verdict_text='✓ Исследуемый метод лучше' if metrics['comparison'][
                    'test_better_by_composite'] else '✗ Базовый метод лучше'
            )

        # Таблица расстояний
        html_content += '''
            <h2>Детальные Расстояния</h2>
            <table class="distance-table">
                <tr>
                    <th>Тип</th>
                    <th>Метка</th>
                    <th>Косинусное расстояние</th>
                    <th>Евклидово расстояние</th>
                    <th>Превышает порог T</th>
                </tr>
    '''

        for dist in distances:
            row_class = "test-row" if dist['source'] == 'test' else "base-row"
            exceeds_threshold = dist['cosine_distance'] > metrics['threshold_T']
            exceeds_text = '<span class="good">Да</span>' if exceeds_threshold else '<span class="bad">Нет</span>'

            html_content += '''
                <tr class="{row_class}">
                    <td>{type_text}</td>
                    <td>{label}</td>
                    <td>{cosine_dist:.4f}</td>
                    <td>{euclidean_dist:.4f}</td>
                    <td>{exceeds_text}</td>
                </tr>
    '''.format(
                row_class=row_class,
                type_text='Исследуемый' if dist['source'] == 'test' else 'Базовый',
                label=dist['label'],
                cosine_dist=dist['cosine_distance'],
                euclidean_dist=dist['euclidean_distance'],
                exceeds_text=exceeds_text
            )

        html_content += '''
            </table>
    '''

        # Итоговый вывод
        html_content += '''
            <div class="summary">
                <h2>Итоговый Вывод</h2>
    '''

        if metrics.get('comparison'):
            if metrics['comparison']['test_better_by_composite']:
                html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="good">превосходные результаты</span> 
                по комплексной оценке. Метод успешно отделяет элементы интерфейса от фона и превосходит базовую модель.</p>
    '''
            else:
                html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="bad">результаты хуже базовой модели</span> 
                по комплексной оценке. Требуется доработка метода.</p>
    '''
        else:
            if metrics.get('test_metrics', {}).get('binary_detection_rate', 0) > 50:
                html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="good">хорошие результаты</span> 
                (Binary Detection Rate > 50%). Модель успешно отделяет элементы интерфейса от фона.</p>
    '''
            else:
                html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="bad">неудовлетворительные результаты</span> 
                (Binary Detection Rate ≤ 50%). Модель плохо отделяет элементы интерфейса от фона.</p>
    '''

        html_content += '''
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
    '''

        with open("benchmark_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print("HTML отчет сохранен в benchmark_report.html")

    def run_benchmark(self):
        """Запуск полного бенчмарк-теста"""
        print("=" * 60)
        print("Запуск бенчмарк-теста эмбеддингов")
        print("=" * 60)

        if self.threshold_T is not None:
            print(f"Используется порог T из конфигурации: {self.threshold_T}")
        else:
            print("Порог T будет вычислен как среднее расстояние")

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

        print(f"\nПорог T: {metrics['threshold_T']:.4f} ({metrics['threshold_source']})")

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
            print(f"\n--- Сравнительный анализ ---")
            print(f"Binary Detection Rate: "
                  f"{'✓' if metrics['comparison']['test_better_by_bdr'] else '✗'} "
                  f"({metrics['comparison']['bdr_improvement_percent']:+.2f}%)")
            print(f"Среднее расстояние: "
                  f"{'✓' if metrics['comparison']['test_better_by_mean'] else '✗'} "
                  f"({metrics['comparison']['mean_improvement_percent']:+.2f}%)")
            print(f"Стандартное отклонение: "
                  f"{'✓' if metrics['comparison']['test_better_by_std'] else '✗'} "
                  f"({metrics['comparison']['std_improvement_percent']:+.2f}%)")

            print(f"\n--- Итоговые выводы ---")
            print(f"Правило большинства (2 из 3): "
                  f"{'✓ Исследуемый метод лучше' if metrics['comparison']['test_better_by_majority'] else '✗ Базовый метод лучше'}")
            print(f"Композитная оценка: {metrics['comparison']['composite_score']:.4f} "
                  f"({'+' if metrics['comparison']['composite_score'] > 0 else ''}{metrics['comparison']['composite_score']:.2%})")
            print(f"Итоговый вердикт: "
                  f"{'✓ Исследуемый метод лучше' if metrics['comparison']['test_better_by_composite'] else '✗ Базовый метод лучше'}")

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