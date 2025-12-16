import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from config import BenchmarkConfig
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')


class RAGMetricsBenchmark:
    def __init__(self, config=BenchmarkConfig):
        self.embeddings = {}
        self.labels = []
        self.sources = []
        self.parent_index = 0
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

        # 2. Загрузка исследуемых эмбеддингов (uied-qwen-2.5-2-images.json)
        with open(self.embeddings_under_research, 'r', encoding='utf-8') as f:
            uied_data = json.load(f)

        uied_embeddings = []
        print(f"Количество исследуемых эмбеддингов: {len(uied_data)}")
        for i, item in enumerate(uied_data):
            if 'text_embedding' in item:
                embedding = np.array(item['text_embedding'])
                uied_embeddings.append(embedding)
                self.labels.append(f'Исследуемый {i + 1}')
                self.sources.append('test')

        self.embeddings['test'] = np.array(uied_embeddings)

        # 3. Загрузка базовых эмбеддингов (base_model_embeddings.json)
        try:
            with open(self.base_embeddings_path, 'r', encoding='utf-8') as f:
                base_data = json.load(f)

            base_embeddings = []
            for i, item in enumerate(base_data):
                if 'text_embedding' in item:
                    embedding = np.array(item['text_embedding'])
                    if len(embedding) > 100:  # Проверка на валидность
                        base_embeddings.append(embedding)
                        self.labels.append(f'Базовый {i + 1}')
                        self.sources.append('base')

            self.embeddings['base'] = np.array(base_embeddings)
        except Exception as e:
            print(f"Ошибка загрузки базовых эмбеддингов: {e}")
            self.embeddings['base'] = np.array([])

        print(f"Родительский эмбеддинг: {self.embeddings['parent'].shape}")
        print(f"Исследуемые эмбеддинги: {self.embeddings['test'].shape}")
        if len(self.embeddings['base']) > 0:
            print(f"Базовые эмбеддинги: {self.embeddings['base'].shape}")

    def align_dimensions(self):
        """Выравнивание размерностей эмбеддингов"""
        print("\nВыравнивание размерностей...")

        # Находим минимальную размерность среди всех эмбеддингов
        parent_dim = self.embeddings['parent'].shape[1]
        test_dim = self.embeddings['test'].shape[1]
        base_dim = self.embeddings['base'].shape[1] if len(self.embeddings['base']) > 0 else float('inf')

        min_dim = min(parent_dim, test_dim, base_dim)
        print(f"Минимальная размерность: {min_dim}")

        # Функция для обрезки или дополнения эмбеддингов
        def adjust_embedding(embedding, target_dim):
            current_dim = embedding.shape[1]
            if current_dim > target_dim:
                return embedding[:, :target_dim]
            elif current_dim < target_dim:
                padding = np.zeros((embedding.shape[0], target_dim - current_dim))
                return np.hstack([embedding, padding])
            return embedding

        # Выравниваем все эмбеддинги
        self.embeddings['parent'] = adjust_embedding(self.embeddings['parent'], min_dim)
        self.embeddings['test'] = adjust_embedding(self.embeddings['test'], min_dim)
        if len(self.embeddings['base']) > 0:
            self.embeddings['base'] = adjust_embedding(self.embeddings['base'], min_dim)

        print("Размерности выровнены успешно")

    def combine_all_embeddings(self):
        """Объединение всех эмбеддингов в одну матрицу"""
        all_embeddings = [self.embeddings['parent'][0]]

        for emb in self.embeddings['test']:
            all_embeddings.append(emb)

        if len(self.embeddings['base']) > 0:
            for emb in self.embeddings['base']:
                all_embeddings.append(emb)

        return np.array(all_embeddings)

    def calculate_retrieval_metrics(self, embeddings_matrix, k_values=[1, 3, 5, 10]):
        """Вычисление метрик ретривеля (Recall@k)"""
        print("\nВычисление метрик ретривеля...")

        # Создаем индексы для разных типов эмбеддингов
        parent_indices = [0]  # Индекс родительского эмбеддинга
        test_indices = list(range(1, 1 + len(self.embeddings['test'])))
        base_indices = list(range(1 + len(self.embeddings['test']),
                                  1 + len(self.embeddings['test']) + len(self.embeddings['base']))) \
            if len(self.embeddings['base']) > 0 else []

        # Вычисляем матрицу расстояний
        n_samples = embeddings_matrix.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = cosine(embeddings_matrix[i], embeddings_matrix[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Функция для вычисления Recall@k для набора индексов
        def calculate_recall_for_indices(indices, source_name):
            results = {k: 0 for k in k_values}
            total = len(indices)

            if total == 0:
                return results

            for idx in indices:
                # Получаем расстояния от текущего эмбеддинга до всех остальных
                distances = distance_matrix[idx]

                # Исключаем сам элемент из рассмотрения
                distances[idx] = np.inf

                # Находим k ближайших соседей для каждого k
                for k in k_values:
                    # Находим индексы k ближайших соседей
                    nearest_indices = np.argsort(distances)[:k]

                    # Проверяем, находится ли родитель среди ближайших соседей
                    if 0 in nearest_indices:  # 0 - индекс родительского эмбеддинга
                        results[k] += 1

            # Преобразуем в проценты
            for k in k_values:
                results[k] = (results[k] / total) * 100 if total > 0 else 0

            return results

        # Вычисляем Recall@k для исследуемых и базовых эмбеддингов
        test_recall = calculate_recall_for_indices(test_indices, "Исследуемые")
        base_recall = calculate_recall_for_indices(base_indices, "Базовые") if base_indices else {}

        # Общие метрики
        all_child_indices = test_indices + base_indices
        overall_recall = calculate_recall_for_indices(all_child_indices, "Все дочерние")

        metrics = {
            'test_recall': test_recall,
            'base_recall': base_recall,
            'overall_recall': overall_recall,
            'k_values': k_values
        }

        # Вывод результатов
        print("\n--- Результаты Recall@k ---")
        for k in k_values:
            print(f"Recall@{k}:")
            print(f"  Исследуемые: {test_recall[k]:.2f}%")
            if base_recall:
                print(f"  Базовые: {base_recall[k]:.2f}%")
            print(f"  Общий: {overall_recall[k]:.2f}%")

        return metrics

    def calculate_clustering_metrics(self, embeddings_matrix):
        """Вычисление метрик кластеризации"""
        print("\nВычисление метрик кластеризации...")

        # Используем только дочерние эмбеддинги для кластеризации
        child_embeddings = embeddings_matrix[1:]  # Исключаем родительский
        child_sources = self.sources[1:]  # Источники для дочерних эмбеддингов

        # Масштабируем данные
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(child_embeddings)

        metrics = {}

        # 1. K-means кластеризация
        print("\n--- K-means кластеризация ---")

        # Определяем оптимальное количество кластеров с помощью силуэтного коэффициента
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []

        # Пробуем разное количество кластеров (от 2 до min(10, количество образцов))
        max_clusters = min(10, len(child_embeddings) - 1)
        cluster_range = range(2, max_clusters + 1)

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_embeddings)

            # Вычисляем метрики
            if len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(scaled_embeddings, cluster_labels)
                calinski = calinski_harabasz_score(scaled_embeddings, cluster_labels)
                davies = davies_bouldin_score(scaled_embeddings, cluster_labels)

                silhouette_scores.append(silhouette)
                calinski_scores.append(calinski)
                davies_scores.append(davies)
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_scores.append(10)  # Максимально плохой результат

        # Находим оптимальное количество кластеров
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_calinski_idx = np.argmax(calinski_scores)
        best_davies_idx = np.argmin(davies_scores)

        best_n_clusters_silhouette = list(cluster_range)[best_silhouette_idx]
        best_n_clusters_calinski = list(cluster_range)[best_calinski_idx]
        best_n_clusters_davies = list(cluster_range)[best_davies_idx]

        print(f"Оптимальное количество кластеров:")
        print(f"  По силуэтному коэффициенту: {best_n_clusters_silhouette}")
        print(f"  По Calinski-Harabasz: {best_n_clusters_calinski}")
        print(f"  По Davies-Bouldin: {best_n_clusters_davies}")

        # Используем силуэтный коэффициент для выбора оптимального количества кластеров
        optimal_clusters = best_n_clusters_silhouette

        # Выполняем кластеризацию с оптимальным количеством кластеров
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(scaled_embeddings)

        # Вычисляем метрики для оптимальной кластеризации
        kmeans_metrics = {
            'n_clusters': optimal_clusters,
            'silhouette_score': silhouette_scores[best_silhouette_idx],
            'calinski_harabasz_score': calinski_scores[best_calinski_idx],
            'davies_bouldin_score': davies_scores[best_davies_idx],
            'inertia': kmeans.inertia_
        }

        print(f"\nМетрики K-means (k={optimal_clusters}):")
        print(f"  Силуэтный коэффициент: {kmeans_metrics['silhouette_score']:.4f}")
        print(f"  Calinski-Harabasz индекс: {kmeans_metrics['calinski_harabasz_score']:.2f}")
        print(f"  Davies-Bouldin индекс: {kmeans_metrics['davies_bouldin_score']:.4f}")
        print(f"  Инерция: {kmeans_metrics['inertia']:.2f}")

        # 2. DBSCAN кластеризация (для сравнения)
        print("\n--- DBSCAN кластеризация ---")
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(scaled_embeddings)
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

            if n_clusters_dbscan > 1:
                dbscan_silhouette = silhouette_score(scaled_embeddings, dbscan_labels)
                dbscan_calinski = calinski_harabasz_score(scaled_embeddings, dbscan_labels)
                dbscan_davies = davies_bouldin_score(scaled_embeddings, dbscan_labels)
            else:
                dbscan_silhouette = 0
                dbscan_calinski = 0
                dbscan_davies = 10

            dbscan_metrics = {
                'n_clusters': n_clusters_dbscan,
                'silhouette_score': dbscan_silhouette,
                'calinski_harabasz_score': dbscan_calinski,
                'davies_bouldin_score': dbscan_davies,
                'n_noise': sum(dbscan_labels == -1)
            }

            print(f"Количество кластеров: {dbscan_metrics['n_clusters']}")
            print(f"Количество шумовых точек: {dbscan_metrics['n_noise']}")
            print(f"Силуэтный коэффициент: {dbscan_metrics['silhouette_score']:.4f}")

        except Exception as e:
            print(f"Ошибка DBSCAN кластеризации: {e}")
            dbscan_metrics = None

        metrics = {
            'kmeans': kmeans_metrics,
            'dbscan': dbscan_metrics,
            'cluster_labels': kmeans_labels.tolist(),
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_scores': davies_scores,
            'cluster_range': list(cluster_range)
        }

        return metrics, scaled_embeddings, child_sources

    def create_retrieval_visualization(self, retrieval_metrics):
        """Создание визуализации для метрик ретривеля"""
        fig = go.Figure()

        k_values = retrieval_metrics['k_values']

        # Добавляем линии для каждого типа эмбеддингов
        fig.add_trace(go.Scatter(
            x=k_values,
            y=[retrieval_metrics['test_recall'][k] for k in k_values],
            mode='lines+markers',
            name='Исследуемые',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))

        if retrieval_metrics['base_recall']:
            fig.add_trace(go.Scatter(
                x=k_values,
                y=[retrieval_metrics['base_recall'][k] for k in k_values],
                mode='lines+markers',
                name='Базовые',
                line=dict(color='green', width=3),
                marker=dict(size=10)
            ))

        fig.add_trace(go.Scatter(
            x=k_values,
            y=[retrieval_metrics['overall_recall'][k] for k in k_values],
            mode='lines+markers',
            name='Общий',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10)
        ))

        fig.update_layout(
            title='Recall@k для поиска родительского элемента',
            xaxis_title='k (количество ближайших соседей)',
            yaxis_title='Recall@k (%)',
            hovermode='x unified',
            height=500,
            width=800,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template='plotly_white'
        )

        fig.update_yaxes(range=[0, 105])

        fig.write_html("retrieval_metrics.html")
        print("Визуализация метрик ретривеля сохранена в retrieval_metrics.html")

        return fig

    def create_clustering_visualization(self, clustering_metrics, scaled_embeddings, child_sources):
        """Создание визуализации для метрик кластеризации"""

        # 1. График выбора оптимального количества кластеров
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Силуэтный коэффициент',
                'Calinski-Harabasz индекс',
                'Davies-Bouldin индекс',
                'Сравнение метрик'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        cluster_range = clustering_metrics['cluster_range']
        silhouette_scores = clustering_metrics['silhouette_scores']
        calinski_scores = clustering_metrics['calinski_scores']
        davies_scores = clustering_metrics['davies_scores']

        # Силуэтный коэффициент
        fig1.add_trace(
            go.Scatter(
                x=cluster_range,
                y=silhouette_scores,
                mode='lines+markers',
                name='Силуэтный коэффициент',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        fig1.add_vline(
            x=clustering_metrics['kmeans']['n_clusters'],
            line_dash="dash",
            line_color="red",
            row=1, col=1
        )

        # Calinski-Harabasz
        fig1.add_trace(
            go.Scatter(
                x=cluster_range,
                y=calinski_scores,
                mode='lines+markers',
                name='Calinski-Harabasz',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        fig1.add_vline(
            x=clustering_metrics['kmeans']['n_clusters'],
            line_dash="dash",
            line_color="red",
            row=1, col=2
        )

        # Davies-Bouldin
        fig1.add_trace(
            go.Scatter(
                x=cluster_range,
                y=davies_scores,
                mode='lines+markers',
                name='Davies-Bouldin',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        fig1.add_vline(
            x=clustering_metrics['kmeans']['n_clusters'],
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )

        # Нормализованное сравнение
        norm_silhouette = (silhouette_scores - np.min(silhouette_scores)) / (
                    np.max(silhouette_scores) - np.min(silhouette_scores) + 1e-10)
        norm_calinski = (calinski_scores - np.min(calinski_scores)) / (
                    np.max(calinski_scores) - np.min(calinski_scores) + 1e-10)
        norm_davies = 1 - (
                    (davies_scores - np.min(davies_scores)) / (np.max(davies_scores) - np.min(davies_scores) + 1e-10))

        fig1.add_trace(
            go.Scatter(
                x=cluster_range,
                y=norm_silhouette,
                mode='lines+markers',
                name='Норм. силуэт',
                line=dict(color='blue', width=2)
            ),
            row=2, col=2
        )
        fig1.add_trace(
            go.Scatter(
                x=cluster_range,
                y=norm_calinski,
                mode='lines+markers',
                name='Норм. Calinski',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )
        fig1.add_trace(
            go.Scatter(
                x=cluster_range,
                y=norm_davies,
                mode='lines+markers',
                name='Норм. Davies',
                line=dict(color='red', width=2)
            ),
            row=2, col=2
        )

        fig1.update_xaxes(title_text="Количество кластеров", row=1, col=1)
        fig1.update_xaxes(title_text="Количество кластеров", row=1, col=2)
        fig1.update_xaxes(title_text="Количество кластеров", row=2, col=1)
        fig1.update_xaxes(title_text="Количество кластеров", row=2, col=2)

        fig1.update_yaxes(title_text="Значение", row=1, col=1)
        fig1.update_yaxes(title_text="Значение", row=1, col=2)
        fig1.update_yaxes(title_text="Значение", row=2, col=1)
        fig1.update_yaxes(title_text="Нормализованное значение", row=2, col=2)

        fig1.update_layout(
            title='Анализ оптимального количества кластеров',
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        fig1.write_html("clustering_analysis.html")
        print("Анализ кластеризации сохранен в clustering_analysis.html")

        # 2. t-SNE визуализация кластеров
        try:
            # Применяем t-SNE для уменьшения размерности до 2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_embeddings) - 1))
            tsne_result = tsne.fit_transform(scaled_embeddings)

            # Создаем DataFrame для визуализации
            import pandas as pd
            tsne_df = pd.DataFrame({
                'x': tsne_result[:, 0],
                'y': tsne_result[:, 1],
                'cluster': clustering_metrics['cluster_labels'],
                'source': child_sources
            })

            fig2 = go.Figure()

            # Добавляем точки по кластерам
            unique_clusters = sorted(set(clustering_metrics['cluster_labels']))
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']

            for i, cluster in enumerate(unique_clusters):
                cluster_mask = tsne_df['cluster'] == cluster
                cluster_data = tsne_df[cluster_mask]

                fig2.add_trace(go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    name=f'Кластер {cluster}',
                    marker=dict(
                        size=10,
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        line=dict(width=1, color='black')
                    ),
                    text=[f"Источник: {src}" for src in cluster_data['source']],
                    hoverinfo='text'
                ))

            fig2.update_layout(
                title='t-SNE визуализация кластеров',
                xaxis_title='t-SNE компонента 1',
                yaxis_title='t-SNE компонента 2',
                height=600,
                width=800,
                template='plotly_white'
            )

            fig2.write_html("tsne_clusters.html")
            print("t-SNE визуализация кластеров сохранена в tsne_clusters.html")

        except Exception as e:
            print(f"Ошибка t-SNE визуализации: {e}")
            fig2 = None

        return fig1, fig2

    def create_comprehensive_report(self, retrieval_metrics, clustering_metrics):
        """Создание комплексного HTML отчета"""
        html_content = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG Метрики - Комплексный отчет</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 2px solid #4CAF50; }
        .section { margin-bottom: 40px; padding: 20px; border-radius: 8px; }
        .retrieval-section { background: #e8f4f8; border-left: 5px solid #2196f3; }
        .clustering-section { background: #f0f8f0; border-left: 5px solid #4caf50; }
        .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: center; }
        .metrics-table th { background-color: #4CAF50; color: white; font-weight: bold; }
        .test-row { background-color: #e8f4fc; }
        .base-row { background-color: #f0f8f0; }
        .overall-row { background-color: #fffacd; }
        .good { color: green; font-weight: bold; }
        .bad { color: red; font-weight: bold; }
        .metric-value { font-weight: bold; color: #2c3e50; }
        h2 { color: #333; margin-top: 30px; }
        h3 { color: #555; margin-top: 20px; }
        .visualization-links { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .summary { background: #2c3e50; color: white; padding: 25px; border-radius: 10px; margin-top: 30px; }
        .highlight-box { background: #fffacd; border: 2px solid #ffeb3b; padding: 15px; border-radius: 5px; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG Метрики - Комплексный отчет</h1>
            <p>Анализ качества эмбеддингов для задач поиска и кластеризации</p>
        </div>
'''

        # Секция метрик ретривеля
        html_content += '''
        <div class="section retrieval-section">
            <h2>📊 Метрики ретривеля (Recall@k)</h2>
            <p>Оценка способности находить родительский элемент среди k ближайших соседей</p>

            <table class="metrics-table">
                <tr>
                    <th>Метрика</th>
        '''

        for k in retrieval_metrics['k_values']:
            html_content += f'<th>Recall@{k}</th>'

        html_content += '''
                </tr>
                <tr class="test-row">
                    <td><strong>Исследуемые эмбеддинги</strong></td>
        '''

        for k in retrieval_metrics['k_values']:
            value = retrieval_metrics['test_recall'][k]
            html_content += f'<td class="metric-value">{value:.2f}%</td>'

        html_content += '''
                </tr>
        '''

        if retrieval_metrics['base_recall']:
            html_content += '''
                <tr class="base-row">
                    <td><strong>Базовые эмбеддинги</strong></td>
            '''

            for k in retrieval_metrics['k_values']:
                value = retrieval_metrics['base_recall'][k]
                html_content += f'<td class="metric-value">{value:.2f}%</td>'

            html_content += '''
                </tr>
            '''

        html_content += '''
                <tr class="overall-row">
                    <td><strong>Общий результат</strong></td>
        '''

        for k in retrieval_metrics['k_values']:
            value = retrieval_metrics['overall_recall'][k]
            html_content += f'<td class="metric-value">{value:.2f}%</td>'

        html_content += '''
                </tr>
            </table>

            <div class="highlight-box">
                <h3>Интерпретация результатов Recall@k:</h3>
                <ul>
                    <li><strong>Recall@1 > 50%</strong>: Отличный результат, родитель находится самым ближайшим соседом</li>
                    <li><strong>Recall@3 > 70%</strong>: Хороший результат, родитель находится в топ-3 соседях</li>
                    <li><strong>Recall@5 > 80%</strong>: Удовлетворительный результат</li>
                    <li><strong>Recall@10 > 90%</strong>: Минимально приемлемый результат</li>
                </ul>
            </div>
        </div>
'''

        # Секция метрик кластеризации
        html_content += '''
        <div class="section clustering-section">
            <h2>🔍 Метрики кластеризации</h2>
            <p>Оценка качества группировки однотипных элементов</p>

            <h3>K-means кластеризация</h3>
            <table class="metrics-table">
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Интерпретация</th>
                </tr>
        '''

        kmeans_metrics = clustering_metrics['kmeans']

        # Силуэтный коэффициент
        silhouette = kmeans_metrics['silhouette_score']
        silhouette_class = 'good' if silhouette > 0.5 else 'bad' if silhouette < 0.3 else ''
        silhouette_interpretation = "Отличное разделение" if silhouette > 0.7 else \
            "Хорошее разделение" if silhouette > 0.5 else \
                "Умеренное разделение" if silhouette > 0.3 else "Слабое разделение"

        html_content += f'''
                <tr>
                    <td>Силуэтный коэффициент</td>
                    <td class="metric-value {silhouette_class}">{silhouette:.4f}</td>
                    <td>{silhouette_interpretation}</td>
                </tr>
        '''

        # Calinski-Harabasz
        calinski = kmeans_metrics['calinski_harabasz_score']
        calinski_class = 'good' if calinski > 100 else 'bad' if calinski < 50 else ''
        calinski_interpretation = "Отличная компактность и разделение" if calinski > 200 else \
            "Хорошая компактность и разделение" if calinski > 100 else \
                "Умеренная компактность" if calinski > 50 else "Слабая компактность"

        html_content += f'''
                <tr>
                    <td>Calinski-Harabasz индекс</td>
                    <td class="metric-value {calinski_class}">{calinski:.2f}</td>
                    <td>{calinski_interpretation}</td>
                </tr>
        '''

        # Davies-Bouldin
        davies = kmeans_metrics['davies_bouldin_score']
        davies_class = 'good' if davies < 0.5 else 'bad' if davies > 1.0 else ''
        davies_interpretation = "Отличное разделение кластеров" if davies < 0.3 else \
            "Хорошее разделение" if davies < 0.5 else \
                "Умеренное разделение" if davies < 1.0 else "Слабое разделение"

        html_content += f'''
                <tr>
                    <td>Davies-Bouldin индекс</td>
                    <td class="metric-value {davies_class}">{davies:.4f}</td>
                    <td>{davies_interpretation}</td>
                </tr>
        '''

        # Количество кластеров
        html_content += f'''
                <tr>
                    <td>Оптимальное количество кластеров</td>
                    <td class="metric-value">{kmeans_metrics['n_clusters']}</td>
                    <td>Определено на основе силуэтного коэффициента</td>
                </tr>
        '''

        html_content += '''
            </table>
'''

        # DBSCAN метрики
        if clustering_metrics['dbscan']:
            dbscan_metrics = clustering_metrics['dbscan']
            html_content += f'''
            <h3>DBSCAN кластеризация</h3>
            <table class="metrics-table">
                <tr>
                    <td>Количество кластеров</td>
                    <td class="metric-value">{dbscan_metrics['n_clusters']}</td>
                </tr>
                <tr>
                    <td>Шумовые точки</td>
                    <td class="metric-value">{dbscan_metrics['n_noise']}</td>
                </tr>
                <tr>
                    <td>Силуэтный коэффициент</td>
                    <td class="metric-value">{dbscan_metrics['silhouette_score']:.4f}</td>
                </tr>
            </table>
'''

        html_content += '''
            <div class="highlight-box">
                <h3>Рекомендации по кластеризации:</h3>
                <ul>
                    <li><strong>Высокий силуэтный коэффициент (>0.7)</strong>: Эмбеддинги хорошо разделяются на кластеры</li>
                    <li><strong>Низкий Davies-Bouldin индекс (<0.5)</strong>: Кластеры компактные и хорошо разделены</li>
                    <li><strong>Оптимальное количество кластеров</strong>: Указывает на естественное группирование данных</li>
                </ul>
            </div>
        </div>
'''

        # Ссылки на визуализации
        html_content += '''
        <div class="visualization-links">
            <h2>📈 Визуализации</h2>
            <p>Интерактивные графики для детального анализа:</p>
            <ul>
                <li><a href="retrieval_metrics.html" target="_blank">📊 Recall@k метрики</a></li>
                <li><a href="clustering_analysis.html" target="_blank">🔍 Анализ кластеризации</a></li>
                <li><a href="tsne_clusters.html" target="_blank">🎨 t-SNE визуализация кластеров</a></li>
            </ul>
        </div>
'''

        # Итоговый вывод
        html_content += '''
        <div class="summary">
            <h2>Итоговый вывод</h2>
'''

        # Анализ ретривеля
        recall_at_1 = retrieval_metrics['overall_recall'][1]
        recall_at_5 = retrieval_metrics['overall_recall'][5]

        if recall_at_1 > 50:
            html_content += '''
            <p><strong>📈 Ретривель:</strong> Отличные результаты! Родительский элемент успешно находится среди ближайших соседей.</p>
'''
        elif recall_at_5 > 70:
            html_content += '''
            <p><strong>📈 Ретривель:</strong> Хорошие результаты. Родитель находится в топ-5 ближайших соседей.</p>
'''
        else:
            html_content += '''
            <p><strong>📈 Ретривель:</strong> Результаты требуют улучшения. Родительский элемент плохо находится среди соседей.</p>
'''

        # Анализ кластеризации
        silhouette = kmeans_metrics['silhouette_score']

        if silhouette > 0.7:
            html_content += '''
            <p><strong>🔍 Кластеризация:</strong> Отличное качество кластеризации! Эмбеддинги хорошо группируются.</p>
'''
        elif silhouette > 0.5:
            html_content += '''
            <p><strong>🔍 Кластеризация:</strong> Хорошее качество кластеризации. Наблюдается четкое разделение на кластеры.</p>
'''
        else:
            html_content += '''
            <p><strong>🔍 Кластеризация:</strong> Качество кластеризации требует улучшения. Эмбеддинги плохо разделяются на кластеры.</p>
'''

        # Общий вердикт
        if recall_at_5 > 70 and silhouette > 0.5:
            html_content += '''
            <p><strong>✅ Общий вердикт:</strong> Эмбеддинги демонстрируют высокое качество для RAG-задач.</p>
'''
        elif recall_at_5 > 50 or silhouette > 0.3:
            html_content += '''
            <p><strong>⚠️ Общий вердикт:</strong> Удовлетворительное качество, но есть возможности для улучшения.</p>
'''
        else:
            html_content += '''
            <p><strong>❌ Общий вердикт:</strong> Качество эмбеддингов требует значительного улучшения для RAG-задач.</p>
'''

        html_content += '''
        </div>
    </div>
</body>
</html>
'''

        with open("rag_metrics_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print("Комплексный отчет сохранен в rag_metrics_report.html")

    def run_benchmark(self):
        """Запуск бенчмарк-теста RAG метрик"""
        print("=" * 60)
        print("Запуск RAG метрик бенчмарк-теста")
        print("=" * 60)

        # Шаг 1: Загрузка эмбеддингов
        self.load_embeddings()

        # Шаг 2: Выравнивание размерностей
        self.align_dimensions()

        # Шаг 3: Объединение всех эмбеддингов
        all_embeddings = self.combine_all_embeddings()
        print(f"\nОбщее количество эмбеддингов: {all_embeddings.shape}")

        # Шаг 4: Вычисление метрик ретривеля
        retrieval_metrics = self.calculate_retrieval_metrics(all_embeddings)

        # Шаг 5: Вычисление метрик кластеризации
        clustering_metrics, scaled_embeddings, child_sources = self.calculate_clustering_metrics(all_embeddings)

        # Шаг 6: Создание визуализаций
        self.create_retrieval_visualization(retrieval_metrics)
        fig1, fig2 = self.create_clustering_visualization(clustering_metrics, scaled_embeddings, child_sources)

        # Шаг 7: Создание комплексного отчета
        self.create_comprehensive_report(retrieval_metrics, clustering_metrics)

        # Вывод результатов в консоль
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ RAG МЕТРИК")
        print("=" * 60)

        print("\n📊 МЕТРИКИ РЕТРИВЕЛЯ:")
        print("-" * 40)
        for k in retrieval_metrics['k_values']:
            print(f"Recall@{k}: {retrieval_metrics['overall_recall'][k]:.2f}%")

        print("\n🔍 МЕТРИКИ КЛАСТЕРИЗАЦИИ:")
        print("-" * 40)
        kmeans_metrics = clustering_metrics['kmeans']
        print(f"Оптимальное количество кластеров: {kmeans_metrics['n_clusters']}")
        print(f"Силуэтный коэффициент: {kmeans_metrics['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz индекс: {kmeans_metrics['calinski_harabasz_score']:.2f}")
        print(f"Davies-Bouldin индекс: {kmeans_metrics['davies_bouldin_score']:.4f}")

        print("\n" + "=" * 60)
        print("СОЗДАННЫЕ ФАЙЛЫ:")
        print("=" * 60)
        print("1. retrieval_metrics.html - Визуализация Recall@k")
        print("2. clustering_analysis.html - Анализ кластеризации")
        print("3. tsne_clusters.html - t-SNE визуализация кластеров")
        print("4. rag_metrics_report.html - Комплексный HTML отчет")
        print("=" * 60)


# Запуск бенчмарк-теста RAG метрик
if __name__ == "__main__":
    benchmark = RAGMetricsBenchmark()
    benchmark.run_benchmark()