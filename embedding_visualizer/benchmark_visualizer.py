import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import warnings
from config import BenchmarkConfig

warnings.filterwarnings('ignore')


class EmbeddingBenchmark:
    def __init__(self, config=BenchmarkConfig):
        self.embeddings = {}
        self.labels = []
        self.sources = []
        self.parent_indices = []
        self.parent_info = []  # Информация о родителях (question, id и т.д.)
        self.core_embedding_path = config.core_embeddings_path
        self.embeddings_under_research = config.embeddings_under_research
        self.base_embeddings_path = config.base_embeddings_path
        self.threshold_T = getattr(config, 'threshold_T', None)

    def load_embeddings(self):
        """Загрузка эмбеддингов из всех файлов"""

        # 1. Загрузка родительских эмбеддингов (core_embedding.json)
        with open(self.core_embedding_path, 'r', encoding='utf-8') as f:
            core_data = json.load(f)

        parent_embeddings = []
        for i, item in enumerate(core_data):
            if 'text_embedding' in item:
                embedding = np.array(item['text_embedding']).reshape(1, -1)
                parent_embeddings.append(embedding)
                self.labels.append(f'Родитель {i + 1}: {item.get("question", "")[:50]}...')
                self.sources.append('parent')
                self.parent_info.append({
                    'id': i,
                    'question': item.get('question', f'Parent_{i + 1}'),
                    'index': len(self.labels) - 1
                })

        self.embeddings['parents'] = np.vstack(parent_embeddings) if parent_embeddings else np.array([])
        self.parent_indices = list(range(len(parent_embeddings)))

        print(f"Загружено родительских эмбеддингов: {len(parent_embeddings)}")
        if len(parent_embeddings) > 0:
            print(f"Размер родительских эмбеддингов: {self.embeddings['parents'].shape}")

        # 2. Загрузка исследуемых эмбеддингов (uied-qwen-2.5-2-images.json)
        with open(self.embeddings_under_research, 'r', encoding='utf-8') as f:
            uied_data = json.load(f)

        uied_embeddings = []
        print(f"Количество исследуемых: {len(uied_data)}")
        for i, item in enumerate(uied_data):
            if 'text_embedding' in item:
                embedding = np.array(item['text_embedding'])
                uied_embeddings.append(embedding)
                self.labels.append(f'Исследуемый {i + 1}')
                self.sources.append('test')

        self.embeddings['test'] = np.array(uied_embeddings) if uied_embeddings else np.array([])
        print(f"Исследуемые эмбеддинги: {self.embeddings['test'].shape}")

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

            self.embeddings['base'] = np.array(base_embeddings) if base_embeddings else np.array([])
            print(f"Базовые эмбеддинги: {self.embeddings['base'].shape}")
        except Exception as e:
            print(f"Ошибка загрузки базовых эмбеддингов: {e}")
            self.embeddings['base'] = np.array([])

    def align_dimensions(self):
        """Выравнивание размерностей эмбеддингов"""
        print("\nВыравнивание размерностей...")

        # Получаем размерности всех эмбеддингов
        parents_dim = self.embeddings['parents'].shape[1] if len(self.embeddings['parents']) > 0 else float('inf')
        test_dim = self.embeddings['test'].shape[1] if len(self.embeddings['test']) > 0 else float('inf')
        base_dim = self.embeddings['base'].shape[1] if len(self.embeddings['base']) > 0 else float('inf')

        # Находим минимальную размерность среди всех эмбеддингов
        min_dim = min(parents_dim, test_dim, base_dim)

        print(f"Размерности: Родители={parents_dim}, Исследуемые={test_dim}, Базовые={base_dim}")
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
        if len(self.embeddings['parents']) > 0:
            print("Выравнивание родительских эмбеддингов...")
            self.embeddings['parents'] = adjust_embedding(self.embeddings['parents'], min_dim)

        if len(self.embeddings['test']) > 0:
            print("Выравнивание исследуемых эмбеддингов...")
            self.embeddings['test'] = adjust_embedding(self.embeddings['test'], min_dim)

        if len(self.embeddings['base']) > 0:
            print("Выравнивание базовых эмбеддингов...")
            self.embeddings['base'] = adjust_embedding(self.embeddings['base'], min_dim)

        print("Размерности выровнены успешно")

        # Проверяем результат
        print(f"\nРезультирующие размерности:")
        if len(self.embeddings['parents']) > 0:
            print(f"  Родители: {self.embeddings['parents'].shape}")
        print(f"  Исследуемые: {self.embeddings['test'].shape}")
        if len(self.embeddings['base']) > 0:
            print(f"  Базовые: {self.embeddings['base'].shape}")

    def combine_all_embeddings(self):
        """Объединение всех эмбеддингов в одну матрицу"""
        all_embeddings = []

        # Добавляем родительские эмбеддинги
        if len(self.embeddings['parents']) > 0:
            all_embeddings.extend(self.embeddings['parents'])

        # Добавляем исследуемые эмбеддинги
        if len(self.embeddings['test']) > 0:
            all_embeddings.extend(self.embeddings['test'])

        # Добавляем базовые эмбеддинги
        if len(self.embeddings['base']) > 0:
            all_embeddings.extend(self.embeddings['base'])

        return np.array(all_embeddings)

    def calculate_distances(self, embeddings_matrix):
        """Вычисление расстояний от каждого эмбеддинга до всех родительских"""
        if len(self.parent_indices) == 0:
            return []

        parent_embeddings = embeddings_matrix[self.parent_indices]
        distances = []

        for i, embedding in enumerate(embeddings_matrix):
            if i in self.parent_indices:
                continue  # Пропускаем самих родителей

            # Вычисляем расстояния до всех родителей
            cos_distances = []
            for j, parent_emb in enumerate(parent_embeddings):
                cos_dist = cosine(parent_emb, embedding)
                cos_distances.append(cos_dist)

            cos_distances = np.array(cos_distances)
            cos_similarities = 1 - cos_distances

            # Находим лучшего родителя
            best_parent_idx = int(np.argmin(cos_distances))
            min_distance = float(cos_distances[best_parent_idx])
            max_distance = float(np.max(cos_distances))
            mean_distance = float(np.mean(cos_distances))
            std_distance = float(np.std(cos_distances))

            # Новые метрики для multi-parent
            confidence = min_distance / mean_distance if mean_distance > 0 else 0
            entropy = float(np.std(cos_distances))

            distances.append({
                'index': i,
                'label': self.labels[i],
                'source': self.sources[i],
                'min_cosine_distance': min_distance,
                'max_cosine_distance': max_distance,
                'mean_cosine_distance': mean_distance,
                'std_cosine_distance': std_distance,
                'best_parent': best_parent_idx,
                'best_parent_similarity': 1 - min_distance,
                'parent_confidence': confidence,
                'parent_entropy': entropy,
                'cosine_distances_to_parents': cos_distances.tolist(),
                'cosine_similarities_to_parents': cos_similarities.tolist()
            })

        return distances

    def interpret_cosine_similarity_for_rag(self, mean_similarity):
        """Интерпретация средней косинусной схожести для RAG"""
        if mean_similarity > 0.8:
            interpretation = "Очень высокая схожесть"
            impact = "Чанки очень близки к родителю. Это хорошо для релевантности, но возможна избыточность информации."
            recommendation = "Проверить на повторяемость/избыточность в чанках. Возможно, нужно увеличить разнообразие."
            rag_effect = "RAG будет получать высокорелевантный контекст, но может страдать от недостатка разнообразной информации."
            color = "green"
        elif mean_similarity >= 0.6:
            interpretation = "Нормальная схожесть"
            impact = "Чанки релевантны родителю с хорошим балансом разнообразия."
            recommendation = "Оптимальный диапазон для большинства задач RAG."
            rag_effect = "RAG будет получать релевантный и достаточно разнообразный контекст для генерации качественных ответов."
            color = "blue"
        elif mean_similarity >= 0.4:
            interpretation = "Низкая схожесть"
            impact = "Чанки умеренно далеки от родителя. Возможны проблемы с релевантностью."
            recommendation = "Рассмотреть пересмотр chunking-стратегии или проверку модели эмбеддингов."
            rag_effect = "RAG может получать нерелевантный контекст, что ухудшит точность и фактологичность ответов."
            color = "orange"
        else:
            interpretation = "Очень низкая схожесть"
            impact = "Чанки семантически далеки от родителя. Вероятен шум и низкая релевантность."
            recommendation = "Требуется пересмотр chunking-стратегии, проверка качества данных и модели эмбеддингов."
            rag_effect = "RAG будет страдать от низкокачественного контекста, что приведет к некорректным или галлюцинированным ответов."
            color = "red"

        return {
            'interpretation': interpretation,
            'impact': impact,
            'recommendation': recommendation,
            'rag_effect': rag_effect,
            'color': color,
            'mean_similarity': mean_similarity,
            'mean_distance': 1 - mean_similarity
        }

    def calculate_metrics(self, distances):
        """Вычисление всех метрик с интерпретацией для RAG"""
        # Разделяем расстояния по источникам
        test_distances = [d for d in distances if d['source'] == 'test']
        base_distances = [d for d in distances if d['source'] == 'base']

        all_cosine_distances = [d['min_cosine_distance'] for d in distances]

        # Используем порог T из конфига, если задан, иначе вычисляем как среднее
        if self.threshold_T is not None:
            threshold_T = float(self.threshold_T)
            threshold_source = "заданный в конфигурации"
        else:
            threshold_T = np.mean(all_cosine_distances) if all_cosine_distances else 0.5
            threshold_source = "вычисленный как среднее"

        metrics = {
            'threshold_T': threshold_T,
            'threshold_source': threshold_source,
            'test_metrics': {},
            'base_metrics': {},
            'comparison': {},
            'rag_interpretation': {},
            'parent_coverage': {},
            'multi_parent_metrics': {}
        }

        # Новые метрики для multi-parent
        if distances:
            # Coverage: сколько родителей было выбрано хотя бы одним чанком
            all_best_parents = [d['best_parent'] for d in distances]
            unique_parents = set(all_best_parents)
            parent_coverage = len(unique_parents) / len(self.parent_indices) * 100 if self.parent_indices else 0

            # Distribution of chunks per parent
            parent_distribution = defaultdict(int)
            for d in distances:
                parent_distribution[d['best_parent']] += 1

            metrics['parent_coverage'] = {
                'coverage_percentage': parent_coverage,
                'covered_parents': len(unique_parents),
                'total_parents': len(self.parent_indices),
                'parent_distribution': dict(parent_distribution)
            }

            # Multi-parent specific metrics
            all_confidences = [d['parent_confidence'] for d in distances]
            all_entropies = [d['parent_entropy'] for d in distances]

            metrics['multi_parent_metrics'] = {
                'mean_confidence': np.mean(all_confidences) if all_confidences else 0,
                'mean_entropy': np.mean(all_entropies) if all_entropies else 0,
                'std_confidence': np.std(all_confidences) if all_confidences else 0,
                'std_entropy': np.std(all_entropies) if all_entropies else 0,
                'interpretation': self.interpret_multi_parent_metrics(
                    np.mean(all_confidences) if all_confidences else 0,
                    np.mean(all_entropies) if all_entropies else 0,
                    parent_coverage
                )
            }

        # Метрики для исследуемых эмбеддингов
        if test_distances:
            test_min_distances = [d['min_cosine_distance'] for d in test_distances]
            test_similarities = [d['best_parent_similarity'] for d in test_distances]
            test_confidences = [d['parent_confidence'] for d in test_distances]
            test_entropies = [d['parent_entropy'] for d in test_distances]

            # Интерпретация для RAG
            mean_test_similarity = np.mean(test_similarities)
            rag_interpretation_test = self.interpret_cosine_similarity_for_rag(mean_test_similarity)

            metrics['test_metrics'] = {
                'mean_min_cosine_distance': np.mean(test_min_distances),
                'mean_cosine_similarity': mean_test_similarity,
                'mean_parent_confidence': np.mean(test_confidences),
                'mean_parent_entropy': np.mean(test_entropies),
                'std_min_cosine_distance': np.std(test_min_distances),
                'std_cosine_similarity': np.std(test_similarities),
                'min_cosine_similarity': np.min(test_similarities),
                'max_cosine_similarity': np.max(test_similarities),
                'count': len(test_distances),
                'rag_interpretation': rag_interpretation_test,
                'best_parent_distribution': self.get_best_parent_distribution(test_distances)
            }

        # Метрики для базовых эмбеддингов
        if base_distances:
            base_min_distances = [d['min_cosine_distance'] for d in base_distances]
            base_similarities = [d['best_parent_similarity'] for d in base_distances]
            base_confidences = [d['parent_confidence'] for d in base_distances]
            base_entropies = [d['parent_entropy'] for d in base_distances]

            # Интерпретация для RAG
            mean_base_similarity = np.mean(base_similarities)
            rag_interpretation_base = self.interpret_cosine_similarity_for_rag(mean_base_similarity)

            metrics['base_metrics'] = {
                'mean_min_cosine_distance': np.mean(base_min_distances),
                'mean_cosine_similarity': mean_base_similarity,
                'mean_parent_confidence': np.mean(base_confidences),
                'mean_parent_entropy': np.mean(base_entropies),
                'std_min_cosine_distance': np.std(base_min_distances),
                'std_cosine_similarity': np.std(base_similarities),
                'min_cosine_similarity': np.min(base_similarities),
                'max_cosine_similarity': np.max(base_similarities),
                'count': len(base_distances),
                'rag_interpretation': rag_interpretation_base,
                'best_parent_distribution': self.get_best_parent_distribution(base_distances)
            }

        # Комплексные сравнительные метрики
        if test_distances and base_distances:
            # Получаем значения метрик
            test_mean_sim = metrics['test_metrics']['mean_cosine_similarity']
            base_mean_sim = metrics['base_metrics']['mean_cosine_similarity']

            test_mean_conf = metrics['test_metrics']['mean_parent_confidence']
            base_mean_conf = metrics['base_metrics']['mean_parent_confidence']

            test_mean_ent = metrics['test_metrics']['mean_parent_entropy']
            base_mean_ent = metrics['base_metrics']['mean_parent_entropy']

            # Вычисляем относительные улучшения/ухудшения
            sim_improvement = test_mean_sim - base_mean_sim
            conf_improvement = test_mean_conf - base_mean_conf
            ent_improvement = test_mean_ent - base_mean_ent

            # Взвешенная оценка
            weights = {
                'similarity': 0.4,
                'confidence': 0.3,
                'entropy': 0.3
            }

            # Нормализуем улучшения
            max_sim = max(abs(test_mean_sim), abs(base_mean_sim), 0.01)
            max_conf = max(abs(test_mean_conf), abs(base_mean_conf), 0.01)
            max_ent = max(abs(test_mean_ent), abs(base_mean_ent), 0.01)

            normalized_sim = sim_improvement / max_sim if max_sim > 0 else 0
            normalized_conf = conf_improvement / max_conf if max_conf > 0 else 0
            normalized_ent = ent_improvement / max_ent if max_ent > 0 else 0

            # Композитный скор
            composite_score = (
                    weights['similarity'] * normalized_sim +
                    weights['confidence'] * normalized_conf +
                    weights['entropy'] * normalized_ent
            )

            metrics['comparison'] = {
                'similarity_improvement': sim_improvement,
                'confidence_improvement': conf_improvement,
                'entropy_improvement': ent_improvement,
                'test_better_by_similarity': test_mean_sim > base_mean_sim,
                'test_better_by_confidence': test_mean_conf < base_mean_conf,  # Меньше confidence лучше
                'test_better_by_entropy': test_mean_ent < base_mean_ent,  # Меньше entropy лучше
                'composite_score': composite_score,
                'test_better_by_composite': composite_score > 0,
                'test_better_by_majority': (
                                                   int(test_mean_sim > base_mean_sim) +
                                                   int(test_mean_conf < base_mean_conf) +
                                                   int(test_mean_ent < base_mean_ent)
                                           ) >= 2
            }

        return metrics, test_distances, base_distances

    def interpret_multi_parent_metrics(self, mean_confidence, mean_entropy, coverage):
        """Интерпретация метрик для multi-parent режима"""
        interpretations = []

        # Интерпретация confidence
        if mean_confidence < 0.7:
            interpretations.append("Высокая уверенность в принадлежности к конкретному родителю")
        elif mean_confidence < 0.9:
            interpretations.append("Умеренная уверенность в принадлежности")
        else:
            interpretations.append("Низкая уверенность - чанки одинаково близки ко многим родителям")

        # Интерпретация entropy
        if mean_entropy < 0.1:
            interpretations.append("Низкая энтропия - четкое разделение между родителями")
        elif mean_entropy < 0.3:
            interpretations.append("Умеренная энтропия - некоторое перекрытие между темами")
        else:
            interpretations.append("Высокая энтропия - значительное перекрытие или шум")

        # Интерпретация coverage
        if coverage > 90:
            interpretations.append("Отличное покрытие - чанки распределены по всем родителям")
        elif coverage > 70:
            interpretations.append("Хорошее покрытие - большинство родителей имеют чанки")
        elif coverage > 50:
            interpretations.append("Удовлетворительное покрытие")
        else:
            interpretations.append("Низкое покрытие - многие родители не представлены")

        return "; ".join(interpretations)

    def get_best_parent_distribution(self, distances):
        """Распределение чанков по родителям"""
        distribution = defaultdict(int)
        for d in distances:
            distribution[d['best_parent']] += 1
        return dict(distribution)

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
        """Создание 3D визуализации PCA с multi-parent поддержкой"""

        # Цветовая палитра для родителей
        parent_colors = [
            'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
            'olive', 'cyan', 'magenta', 'yellow', 'teal', 'coral', 'indigo'
        ]

        # Формы маркеров для родителей
        parent_markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up',
                          'triangle-down', 'triangle-left', 'triangle-right', 'pentagon']

        colors = []
        sizes = []
        hover_texts = []
        marker_symbols = []

        for i, source in enumerate(self.sources):
            if source == 'parent':
                # Родители - разные цвета и формы
                parent_idx = self.parent_indices.index(i) if i in self.parent_indices else 0
                colors.append(parent_colors[parent_idx % len(parent_colors)])
                marker_symbols.append(parent_markers[parent_idx % len(parent_markers)])
                sizes.append(20)
                parent_info = self.parent_info[parent_idx] if parent_idx < len(self.parent_info) else {}
                hover_texts.append(
                    f"{self.labels[i]}<br>"
                    f"Родительский элемент<br>"
                    f"Вопрос: {parent_info.get('question', 'N/A')}"
                )
            elif source == 'test' or source == 'base':
                # Для чанков определяем цвет по best_parent
                dist_info = next((d for d in distances if d['index'] == i), None)
                if dist_info:
                    parent_idx = dist_info['best_parent']
                    colors.append(parent_colors[parent_idx % len(parent_colors)])
                    marker_symbols.append('circle')

                    similarity_info = ""
                    if len(dist_info['cosine_similarities_to_parents']) <= 5:
                        sim_text = ", ".join([f"{s:.3f}" for s in dist_info['cosine_similarities_to_parents']])
                        similarity_info = f"<br>Схожести ко всем родителям: [{sim_text}]"

                    hover_texts.append(
                        f"{self.labels[i]}<br>"
                        f"Тип: {source}<br>"
                        f"Лучший родитель: {parent_idx}<br>"
                        f"Мин. косинусное расстояние: {dist_info['min_cosine_distance']:.4f}<br>"
                        f"Косинусная схожесть: {dist_info['best_parent_similarity']:.4f}<br>"
                        f"Confidence: {dist_info['parent_confidence']:.4f}<br>"
                        f"Entropy: {dist_info['parent_entropy']:.4f}"
                        f"{similarity_info}"
                    )
                else:
                    colors.append('gray')
                    marker_symbols.append('circle')
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
                symbol=marker_symbols,
                opacity=0.8,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Эмбеддинги'
        ))

        # Настройки layout
        fig.update_layout(
            title='3D PCA Визуализация Эмбеддингов (Multi-Parent Mode)',
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

        # Добавляем легенду для родителей
        for i in range(len(self.parent_indices)):
            if i < len(parent_colors) and i < len(parent_markers):
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=parent_colors[i],
                        symbol=parent_markers[i]
                    ),
                    name=f'Родитель {i + 1}',
                    showlegend=True
                ))

        # Добавляем легенду для типов
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle'),
            name='Исследуемые',
            showlegend=True
        ))

        if any(s == 'base' for s in self.sources):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color='green', symbol='circle'),
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
        test_distances = [d['min_cosine_distance'] for d in distances if d['source'] == 'test']
        base_distances = [d['min_cosine_distance'] for d in distances if d['source'] == 'base']

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

        # Добавляем вертикальные линии для интерпретации RAG
        fig.add_vrect(
            x0=0, x1=0.2,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Высокая схожесть (>0.8)",
            annotation_position="top left"
        )

        fig.add_vrect(
            x0=0.2, x1=0.4,
            fillcolor="blue", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Нормальная схожесть (0.6-0.8)",
            annotation_position="top left"
        )

        fig.add_vrect(
            x0=0.4, x1=0.6,
            fillcolor="orange", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Низкая схожесть (0.4-0.6)",
            annotation_position="top left"
        )

        fig.add_vrect(
            x0=0.6, x1=1.0,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Очень низкая схожесть (<0.4)",
            annotation_position="top left"
        )

        fig.update_layout(
            title='Распределение Минимальных Косинусных Расстояний до Родительских Эмбеддингов',
            xaxis_title='Минимальное косинусное расстояние (1 - similarity)',
            yaxis_title='Частота',
            barmode='overlay',
            height=500,
            width=800
        )

        fig.write_html("distance_distribution.html")
        print("Распределение расстояний сохранено в distance_distribution.html")

        return fig

    def create_confidence_entropy_plot(self, metrics):
        """Создание графика confidence vs entropy"""
        fig = go.Figure()

        # Данные для исследуемых эмбеддингов
        if metrics.get('test_distances'):
            test_confidences = [d['parent_confidence'] for d in metrics['test_distances']]
            test_entropies = [d['parent_entropy'] for d in metrics['test_distances']]

            fig.add_trace(go.Scatter(
                x=test_confidences,
                y=test_entropies,
                mode='markers',
                name='Исследуемые',
                marker=dict(color='blue', size=10, opacity=0.6),
                text=[f"Чанк {i + 1}" for i in range(len(test_confidences))]
            ))

        # Данные для базовых эмбеддингов
        if metrics.get('base_distances'):
            base_confidences = [d['parent_confidence'] for d in metrics['base_distances']]
            base_entropies = [d['parent_entropy'] for d in metrics['base_distances']]

            fig.add_trace(go.Scatter(
                x=base_confidences,
                y=base_entropies,
                mode='markers',
                name='Базовые',
                marker=dict(color='green', size=10, opacity=0.6),
                text=[f"Чанк {i + 1}" for i in range(len(base_confidences))]
            ))

        # Добавляем линии-ориентиры
        fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                      annotation_text="Confidence=0.7", annotation_position="top right")

        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                      annotation_text="Entropy=0.1", annotation_position="top right")

        fig.update_layout(
            title='Confidence vs Entropy (Multi-Parent Analysis)',
            xaxis_title='Parent Confidence (меньше = лучше)',
            yaxis_title='Parent Entropy (меньше = лучше)',
            height=600,
            width=800,
            showlegend=True
        )

        fig.write_html("confidence_entropy_plot.html")
        print("График confidence vs entropy сохранен в confidence_entropy_plot.html")

        return fig

    def create_rag_interpretation_section(self, metrics):
        """Создание раздела с интерпретацией для RAG"""
        html_content = '''
        <div class="rag-interpretation">
            <h2>Интерпретация для RAG (Retrieval-Augmented Generation) - Multi-Parent Mode</h2>

            <div class="rag-explanation">
                <h3>Что означает multi-parent подход?</h3>
                <p>Вместо одного "родительского" эмбеддинга используется несколько смысловых якорей (anchors), каждый из которых представляет:</p>
                <ul>
                    <li>Отдельный документ или раздел</li>
                    <li>Конкретный intent или запрос</li>
                    <li>Семантическую тему или категорию</li>
                    <li>"Золотой" query для retrieval</li>
                </ul>
                <p>Чанк считается качественным, если он уверенно принадлежит к одному из родителей (низкая энтропия, высокая confidence).</p>
            </div>

            <div class="rag-metrics-explanation">
                <h3>Новые метрики в multi-parent режиме</h3>
                <table class="rag-table">
                    <tr>
                        <th>Метрика</th>
                        <th>Формула/Описание</th>
                        <th>Идеальное значение</th>
                        <th>Интерпретация для RAG</th>
                    </tr>
                    <tr>
                        <td><strong>Parent Confidence</strong></td>
                        <td>min_dist / mean_dist</td>
                        <td>< 0.7</td>
                        <td>Чанк четко принадлежит одному смыслу, не "размазан"</td>
                    </tr>
                    <tr>
                        <td><strong>Parent Entropy</strong></td>
                        <td>std(расстояний до всех родителей)</td>
                        <td>< 0.1</td>
                        <td>Низкая неопределенность, четкое разделение тем</td>
                    </tr>
                    <tr>
                        <td><strong>Parent Coverage</strong></td>
                        <td>% родителей с хотя бы одним чанком</td>
                        <td>> 90%</td>
                        <td>Хорошее покрытие всех смысловых якорей</td>
                    </tr>
                    <tr>
                        <td><strong>Best Parent Similarity</strong></td>
                        <td>1 - min_distance</td>
                        <td>> 0.6</td>
                        <td>Высокая схожесть с ближайшим родителем</td>
                    </tr>
                </table>
            </div>

            <div class="rag-impact">
                <h3>Как это влияет на качество RAG?</h3>
                <ul>
                    <li><strong>Высокий Confidence + низкий Entropy</strong>: Чанки четко привязаны к конкретным смыслам. 
                    RAG получает релевантный и фокусированный контекст.</li>
                    <li><strong>Низкий Confidence (близко к 1.0)</strong>: Чанки одинаково близки ко многим родителям.
                    Это создает шум и снижает точность retrieval.</li>
                    <li><strong>Высокий Entropy</strong>: Чанки "размазаны" по разным темам. 
                    Может указывать на плохой chunking или пересекающиеся темы.</li>
                    <li><strong>Низкий Coverage</strong>: Некоторые родители не имеют чанков.
                    Значит, часть информации не будет доступна для retrieval.</li>
                </ul>
            </div>
        '''

        # Добавляем интерпретацию multi-parent метрик
        if metrics.get('multi_parent_metrics'):
            multi_metrics = metrics['multi_parent_metrics']
            html_content += f'''
            <div class="multi-parent-results" style="border: 2px solid #4CAF50; padding: 15px; margin: 20px 0; border-radius: 10px;">
                <h3>Multi-Parent Метрики (общие)</h3>
                <div class="metric-item">
                    Средний Confidence: <span class="metric-value">{multi_metrics['mean_confidence']:.4f}</span>
                </div>
                <div class="metric-item">
                    Средняя Entropy: <span class="metric-value">{multi_metrics['mean_entropy']:.4f}</span>
                </div>
                <div class="metric-item">
                    <strong>Интерпретация:</strong> {multi_metrics['interpretation']}
                </div>
            </div>
            '''

        # Добавляем coverage информацию
        if metrics.get('parent_coverage'):
            coverage = metrics['parent_coverage']
            html_content += f'''
            <div class="coverage-results" style="border: 2px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 10px;">
                <h3>Parent Coverage</h3>
                <div class="metric-item">
                    Покрытие родителей: <span class="metric-value">{coverage['coverage_percentage']:.1f}%</span>
                </div>
                <div class="metric-item">
                    Покрыто родителей: {coverage['covered_parents']} из {coverage['total_parents']}
                </div>
                <div class="metric-item">
                    <strong>Распределение чанков по родителям:</strong>
                    <ul>
            '''

            for parent_id, count in coverage['parent_distribution'].items():
                html_content += f'<li>Родитель {parent_id + 1}: {count} чанков</li>'

            html_content += '''
                    </ul>
                </div>
            </div>
            '''

        # Добавляем интерпретацию для исследуемых эмбеддингов
        if metrics.get('test_metrics') and 'rag_interpretation' in metrics['test_metrics']:
            rag_test = metrics['test_metrics']['rag_interpretation']
            html_content += f'''
            <div class="rag-results" style="border: 2px solid {rag_test['color']}; padding: 15px; margin: 20px 0; border-radius: 10px;">
                <h3>Интерпретация для исследуемых эмбеддингов</h3>
                <div class="metric-item">
                    Средняя схожесть с лучшим родителем: <span class="metric-value">{rag_test['mean_similarity']:.4f}</span>
                </div>
                <div class="metric-item">
                    <strong>Интерпретация:</strong> <span style="color: {rag_test['color']};">{rag_test['interpretation']}</span>
                </div>
                <div class="metric-item">
                    <strong>Влияние на RAG:</strong> {rag_test['rag_effect']}
                </div>
                <div class="metric-item">
                    <strong>Рекомендация:</strong> {rag_test['recommendation']}
                </div>
            </div>
            '''

        html_content += '''
            <div class="rag-references">
                <h3>Источники и пояснения</h3>
                <ul>
                    <li><strong>Multi-parent подход</strong>: Более реалистичная модель для retrieval, 
                    где у каждого query есть несколько релевантных контекстов (источник: arXiv:2305.14627)</li>
                    <li><strong>Parent Confidence</strong>: Показывает, насколько чанк "уверен" в принадлежности к конкретному родителю. 
                    Низкие значения (0.3-0.7) оптимальны (источник: milvus.io)</li>
                    <li><strong>Parent Entropy</strong>: Измеряет неопределенность принадлежности. 
                    Низкая энтропия указывает на четкое семантическое разделение (источник: information theory)</li>
                    <li><strong>Coverage</strong>: Важная метрика для оценки полноты chunking-стратегии. 
                    Хороший coverage означает, что все темы представлены в чанках (источник: RAG evaluation literature)</li>
                </ul>
                <p><em>Примечание: В multi-parent режиме важна не только схожесть с одним родителем, 
                но и дифференциация между разными родителями.</em></p>
            </div>
        </div>
        '''

        return html_content

    def create_metrics_report(self, metrics, distances):
            """Создание HTML отчета с метриками"""
            # Используем двойные фигурные скобки для экранирования в CSS
            html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Бенчмарк Эмбеддингов - Multi-Parent Отчет</title>
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
            .rag-interpretation {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4CAF50; }}
            .rag-explanation {{ margin: 15px 0; }}
            .rag-impact {{ margin: 15px 0; }}
            .rag-thresholds {{ margin: 15px 0; }}
            .rag-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            .rag-table th, .rag-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            .rag-table th {{ background-color: #4CAF50; color: white; }}
            .rag-results {{ margin: 20px 0; }}
            .rag-references {{ margin-top: 20px; padding: 15px; background: #f0f8ff; border-radius: 8px; }}
            .similarity-badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 0.9em;
                font-weight: bold;
                margin-left: 10px;
            }}
            .similarity-high {{ background: #d4edda; color: #155724; }}
            .similarity-good {{ background: #d1ecf1; color: #0c5460; }}
            .similarity-low {{ background: #fff3cd; color: #856404; }}
            .similarity-poor {{ background: #f8d7da; color: #721c24; }}
            .multi-parent-section {{ background: #f0f8ff; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #2196F3; }}
            .coverage-section {{ background: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .confidence-good {{ color: #2E7D32; }}
            .confidence-bad {{ color: #C62828; }}
            .entropy-good {{ color: #2E7D32; }}
            .entropy-bad {{ color: #C62828; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Бенчмарк Эмбеддингов - Multi-Parent Анализ</h1>
                <p>Сравнение исследуемых и базовых эмбеддингов с несколькими родительскими якорями</p>
                <p><strong>Режим:</strong> Multiple Parent Embeddings ({parent_count} родителей)</p>
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
                parent_count=len(self.parent_indices),
                threshold_T=metrics['threshold_T'],
                threshold_source=metrics['threshold_source']
            )

            # Multi-parent метрики
            if metrics.get('multi_parent_metrics'):
                multi_metrics = metrics['multi_parent_metrics']
                conf_class = 'confidence-good' if multi_metrics['mean_confidence'] < 0.7 else 'confidence-bad'
                conf_text = 'Хорошо' if multi_metrics['mean_confidence'] < 0.7 else 'Плохо'
                ent_class = 'entropy-good' if multi_metrics['mean_entropy'] < 0.1 else 'entropy-bad'
                ent_text = 'Хорошо' if multi_metrics['mean_entropy'] < 0.1 else 'Плохо'

                html_content += '''
            <div class="multi-parent-section">
                <h2>Multi-Parent Метрики</h2>
                <div class="metric-item">
                    Средний Confidence: <span class="metric-value {conf_class}">{mean_conf:.4f}</span>
                    <span class="similarity-badge">{conf_text}</span>
                </div>
                <div class="metric-item">
                    Средняя Entropy: <span class="metric-value {ent_class}">{mean_ent:.4f}</span>
                    <span class="similarity-badge">{ent_text}</span>
                </div>
                <div class="metric-item">
                    <strong>Интерпретация:</strong> {interpretation}
                </div>
            </div>
    '''.format(
                    mean_conf=multi_metrics['mean_confidence'],
                    conf_class=conf_class,
                    conf_text=conf_text,
                    mean_ent=multi_metrics['mean_entropy'],
                    ent_class=ent_class,
                    ent_text=ent_text,
                    interpretation=multi_metrics['interpretation']
                )

            # Coverage информация
            if metrics.get('parent_coverage'):
                coverage = metrics['parent_coverage']
                coverage_class = 'good' if coverage['coverage_percentage'] > 70 else 'bad' if coverage[
                                                                                                  'coverage_percentage'] < 50 else 'neutral'
                html_content += '''
            <div class="coverage-section">
                <h2>Parent Coverage</h2>
                <div class="metric-item">
                    Покрытие родителей: <span class="metric-value {cov_class}">{coverage_percentage:.1f}%</span>
                </div>
                <div class="metric-item">
                    Покрыто родителей: {covered_parents} из {total_parents}
                </div>
                <div class="metric-item">
                    <strong>Распределение чанков по родителям:</strong>
                    <table class="comparison-table" style="width: 50%;">
                        <tr>
                            <th>Родитель</th>
                            <th>Количество чанков</th>
                        </tr>
    '''.format(
                    coverage_percentage=coverage['coverage_percentage'],
                    cov_class=coverage_class,
                    covered_parents=coverage['covered_parents'],
                    total_parents=coverage['total_parents']
                )

                for parent_id, count in coverage['parent_distribution'].items():
                    html_content += '''
                        <tr>
                            <td>Родитель {parent_id}</td>
                            <td>{count}</td>
                        </tr>
    '''.format(parent_id=parent_id + 1, count=count)

                html_content += '''
                    </table>
                </div>
            </div>
    '''

            # Основные метрики
            html_content += '''
            <div class="metrics">
                <h2>Основные Метрики</h2>
    '''

            if metrics['test_metrics']:
                test_mean_sim = metrics['test_metrics']['mean_cosine_similarity']
                test_sim_badge = 'similarity-high' if test_mean_sim > 0.8 else 'similarity-good' if test_mean_sim >= 0.6 else 'similarity-low' if test_mean_sim >= 0.4 else 'similarity-poor'

                test_conf = metrics['test_metrics']['mean_parent_confidence']
                test_conf_badge = 'confidence-good' if test_conf < 0.7 else 'confidence-bad'
                test_conf_text = 'Хорошо' if test_conf < 0.7 else 'Плохо'

                test_ent = metrics['test_metrics']['mean_parent_entropy']
                test_ent_badge = 'entropy-good' if test_ent < 0.1 else 'entropy-bad'
                test_ent_text = 'Хорошо' if test_ent < 0.1 else 'Плохо'

                html_content += '''
                <h3>Исследуемые эмбеддинги ({test_count} шт.)</h3>
                <div class="metric-item">
                    Средняя схожесть с лучшим родителем: <span class="metric-value">{test_sim:.4f}</span>
                    <span class="similarity-badge {sim_badge}">{sim_text}</span>
                </div>
                <div class="metric-item">
                    Средний Confidence: <span class="metric-value {conf_class}">{test_conf:.4f}</span>
                    <span class="similarity-badge">{conf_text}</span>
                </div>
                <div class="metric-item">
                    Средняя Entropy: <span class="metric-value {ent_class}">{test_ent:.4f}</span>
                    <span class="similarity-badge">{ent_text}</span>
                </div>
                <div class="metric-item">
                    Стандартное отклонение схожести: <span class="metric-value">{test_std:.4f}</span>
                </div>
                <div class="metric-item">
                    Диапазон схожести: [{test_min_sim:.4f}, {test_max_sim:.4f}]
                </div>
    '''.format(
                    test_count=metrics['test_metrics']['count'],
                    test_sim=test_mean_sim,
                    sim_badge=test_sim_badge,
                    sim_text='Высокая' if test_mean_sim > 0.8 else 'Нормальная' if test_mean_sim >= 0.6 else 'Низкая' if test_mean_sim >= 0.4 else 'Очень низкая',
                    test_conf=test_conf,
                    conf_class=test_conf_badge,
                    conf_text=test_conf_text,
                    test_ent=test_ent,
                    ent_class=test_ent_badge,
                    ent_text=test_ent_text,
                    test_std=metrics['test_metrics']['std_cosine_similarity'],
                    test_min_sim=metrics['test_metrics']['min_cosine_similarity'],
                    test_max_sim=metrics['test_metrics']['max_cosine_similarity']
                )

            if metrics['base_metrics']:
                base_mean_sim = metrics['base_metrics']['mean_cosine_similarity']
                base_sim_badge = 'similarity-high' if base_mean_sim > 0.8 else 'similarity-good' if base_mean_sim >= 0.6 else 'similarity-low' if base_mean_sim >= 0.4 else 'similarity-poor'

                base_conf = metrics['base_metrics']['mean_parent_confidence']
                base_conf_badge = 'confidence-good' if base_conf < 0.7 else 'confidence-bad'
                base_conf_text = 'Хорошо' if base_conf < 0.7 else 'Плохо'

                base_ent = metrics['base_metrics']['mean_parent_entropy']
                base_ent_badge = 'entropy-good' if base_ent < 0.1 else 'entropy-bad'
                base_ent_text = 'Хорошо' if base_ent < 0.1 else 'Плохо'

                html_content += '''
                <h3>Базовые эмбеддинги ({base_count} шт.)</h3>
                <div class="metric-item">
                    Средняя схожесть с лучшим родителем: <span class="metric-value">{base_sim:.4f}</span>
                    <span class="similarity-badge {sim_badge}">{sim_text}</span>
                </div>
                <div class="metric-item">
                    Средний Confidence: <span class="metric-value {conf_class}">{base_conf:.4f}</span>
                    <span class="similarity-badge">{conf_text}</span>
                </div>
                <div class="metric-item">
                    Средняя Entropy: <span class="metric-value {ent_class}">{base_ent:.4f}</span>
                    <span class="similarity-badge">{ent_text}</span>
                </div>
                <div class="metric-item">
                    Стандартное отклонение схожести: <span class="metric-value">{base_std:.4f}</span>
                </div>
                <div class="metric-item">
                    Диапазон схожести: [{base_min_sim:.4f}, {base_max_sim:.4f}]
                </div>
    '''.format(
                    base_count=metrics['base_metrics']['count'],
                    base_sim=base_mean_sim,
                    sim_badge=base_sim_badge,
                    sim_text='Высокая' if base_mean_sim > 0.8 else 'Нормальная' if base_mean_sim >= 0.6 else 'Низкая' if base_mean_sim >= 0.4 else 'Очень низкая',
                    base_conf=base_conf,
                    conf_class=base_conf_badge,
                    conf_text=base_conf_text,
                    base_ent=base_ent,
                    ent_class=base_ent_badge,
                    ent_text=base_ent_text,
                    base_std=metrics['base_metrics']['std_cosine_similarity'],
                    base_min_sim=metrics['base_metrics']['min_cosine_similarity'],
                    base_max_sim=metrics['base_metrics']['max_cosine_similarity']
                )

            html_content += '''
            </div>
    '''

            # Сравнительные метрики
            if metrics.get('comparison'):
                comp = metrics['comparison']
                html_content += '''
            <div class="comparison">
                <h2>Сравнительный анализ</h2>

                <table class="comparison-table">
                    <tr>
                        <th>Метрика</th>
                        <th>Исследуемый метод</th>
                        <th>Базовый метод</th>
                        <th>Улучшение</th>
                        <th>Результат</th>
                    </tr>
                    <tr>
                        <td>Средняя схожесть</td>
                        <td>{test_sim:.4f}</td>
                        <td>{base_sim:.4f}</td>
                        <td class="{sim_imp_class}">{sim_sign}{sim_imp:.4f}</td>
                        <td class="{sim_res_class}">{sim_res_text}</td>
                    </tr>
                    <tr>
                        <td>Confidence</td>
                        <td>{test_conf:.4f}</td>
                        <td>{base_conf:.4f}</td>
                        <td class="{conf_imp_class}">{conf_sign}{conf_imp:.4f}</td>
                        <td class="{conf_res_class}">{conf_res_text}</td>
                    </tr>
                    <tr>
                        <td>Entropy</td>
                        <td>{test_ent:.4f}</td>
                        <td>{base_ent:.4f}</td>
                        <td class="{ent_imp_class}">{ent_sign}{ent_imp:.4f}</td>
                        <td class="{ent_res_class}">{ent_res_text}</td>
                    </tr>
                </table>

                <div class="metric-item">
                    Правило большинства (2 из 3): <span class="{majority_class}">{majority_text}</span>
                </div>
                <div class="metric-item">
                    Композитная оценка: <span class="{comp_class}">{composite_score:.4f}</span>
                </div>
                <div class="metric-item">
                    Итоговый вердикт: <span class="{verdict_class}">{verdict_text}</span>
                </div>
            </div>
    '''.format(
                    test_sim=metrics['test_metrics']['mean_cosine_similarity'],
                    base_sim=metrics['base_metrics']['mean_cosine_similarity'],
                    sim_imp=comp['similarity_improvement'],
                    sim_imp_class='good' if comp['similarity_improvement'] > 0 else 'bad',
                    sim_sign='+' if comp['similarity_improvement'] > 0 else '',
                    sim_res_class='good' if comp['test_better_by_similarity'] else 'bad',
                    sim_res_text='✓ Лучше' if comp['test_better_by_similarity'] else '✗ Хуже',

                    test_conf=metrics['test_metrics']['mean_parent_confidence'],
                    base_conf=metrics['base_metrics']['mean_parent_confidence'],
                    conf_imp=comp['confidence_improvement'],
                    conf_imp_class='good' if comp['confidence_improvement'] < 0 else 'bad',
                    conf_sign='' if comp['confidence_improvement'] < 0 else '+',
                    conf_res_class='good' if comp['test_better_by_confidence'] else 'bad',
                    conf_res_text='✓ Лучше' if comp['test_better_by_confidence'] else '✗ Хуже',

                    test_ent=metrics['test_metrics']['mean_parent_entropy'],
                    base_ent=metrics['base_metrics']['mean_parent_entropy'],
                    ent_imp=comp['entropy_improvement'],
                    ent_imp_class='good' if comp['entropy_improvement'] < 0 else 'bad',
                    ent_sign='' if comp['entropy_improvement'] < 0 else '+',
                    ent_res_class='good' if comp['test_better_by_entropy'] else 'bad',
                    ent_res_text='✓ Лучше' if comp['test_better_by_entropy'] else '✗ Хуже',

                    majority_class='good' if comp['test_better_by_majority'] else 'bad',
                    majority_text='✓ Исследуемый метод лучше' if comp[
                        'test_better_by_majority'] else '✗ Базовый метод лучше',
                    composite_score=comp['composite_score'],
                    comp_class='good' if comp['composite_score'] > 0 else 'bad',
                    verdict_class='good' if comp['test_better_by_composite'] else 'bad',
                    verdict_text='✓ Исследуемый метод лучше' if comp[
                        'test_better_by_composite'] else '✗ Базовый метод лучше'
                )

            # Добавляем раздел с интерпретацией для RAG
            html_content += self.create_rag_interpretation_section(metrics)

            # Таблица расстояний
            html_content += '''
            <h2>Детальные Расстояния (первые 20 записей)</h2>
            <table class="distance-table">
                <tr>
                    <th>Тип</th>
                    <th>Метка</th>
                    <th>Лучший родитель</th>
                    <th>Мин. косинусное расстояние</th>
                    <th>Косинусная схожесть</th>
                    <th>Confidence</th>
                    <th>Entropy</th>
                    <th>Превышает порог T</th>
                </tr>
    '''

            for i, dist in enumerate(distances[:20]):  # Показываем только первые 20 для читаемости
                if i >= 20:
                    break

                row_class = "test-row" if dist['source'] == 'test' else "base-row"
                exceeds_threshold = dist['min_cosine_distance'] > metrics['threshold_T']
                exceeds_text = '<span class="good">Да</span>' if exceeds_threshold else '<span class="bad">Нет</span>'

                # Определяем цвет для схожести
                similarity = dist['best_parent_similarity']
                sim_color = 'green' if similarity > 0.8 else 'blue' if similarity >= 0.6 else 'orange' if similarity >= 0.4 else 'red'

                # Определяем цвет для confidence
                conf_color = '#2E7D32' if dist['parent_confidence'] < 0.7 else '#C62828'

                # Определяем цвет для entropy
                ent_color = '#2E7D32' if dist['parent_entropy'] < 0.1 else '#C62828'

                html_content += '''
                <tr class="{row_class}">
                    <td>{type_text}</td>
                    <td>{label}</td>
                    <td>{best_parent}</td>
                    <td>{min_cosine_dist:.4f}</td>
                    <td><span style="color: {sim_color};">{cosine_sim:.4f}</span></td>
                    <td><span style="color: {conf_color};">{confidence:.4f}</span></td>
                    <td><span style="color: {ent_color};">{entropy:.4f}</span></td>
                    <td>{exceeds_text}</td>
                </tr>
    '''.format(
                    row_class=row_class,
                    type_text='Исследуемый' if dist['source'] == 'test' else 'Базовый',
                    label=dist['label'],
                    best_parent=dist['best_parent'] + 1,
                    min_cosine_dist=dist['min_cosine_distance'],
                    cosine_sim=similarity,
                    sim_color=sim_color,
                    confidence=dist['parent_confidence'],
                    conf_color=conf_color,
                    entropy=dist['parent_entropy'],
                    ent_color=ent_color,
                    exceeds_text=exceeds_text
                )

            html_content += '''
            </table>
            <p><em>Показано {shown} из {total} записей. Полные данные доступны в JSON отчете.</em></p>
    '''.format(shown=min(20, len(distances)), total=len(distances))

            # Итоговый вывод
            html_content += '''
            <div class="summary">
                <h2>Итоговый Вывод (Multi-Parent Mode)</h2>
    '''

            if metrics.get('comparison'):
                if metrics['comparison']['test_better_by_composite']:
                    html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="good">превосходные результаты</span> 
                в multi-parent режиме. Метод успешно дифференцирует чанки по различным смысловым якорям.</p>
    '''
                else:
                    html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="bad">результаты хуже базовой модели</span> 
                в multi-parent режиме. Требуется доработка метода для лучшего разделения смыслов.</p>
    '''
            else:
                if metrics.get('test_metrics', {}).get('mean_cosine_similarity', 0) > 0.6:
                    html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="good">хорошие результаты</span> 
                (средняя схожесть > 0.6). Чанки релевантны родительским якорям.</p>
    '''
                else:
                    html_content += '''
                <p>Исследуемые эмбеддинги показывают <span class="bad">неудовлетворительные результаты</span> 
                (средняя схожесть ≤ 0.6). Требуется улучшение качества эмбеддингов или chunking-стратегии.</p>
    '''

            # Добавляем рекомендации по multi-parent метрикам
            if metrics.get('multi_parent_metrics'):
                multi_metrics = metrics['multi_parent_metrics']
                html_content += '''
                <p><strong>Анализ multi-parent метрик:</strong> {}</p>
    '''.format(multi_metrics['interpretation'])

            if metrics.get('parent_coverage'):
                coverage = metrics['parent_coverage']
                if coverage['coverage_percentage'] < 70:
                    html_content += '''
                <p><strong>Рекомендация по coverage:</strong> Только {covered_parents} из {total_parents} 
                родителей имеют чанки. Рассмотрите пересмотр chunking-стратегии для лучшего покрытия всех тем.</p>
    '''.format(covered_parents=coverage['covered_parents'], total_parents=coverage['total_parents'])

            html_content += '''
            </div>

            <div style="margin-top: 40px; text-align: center;">
                <p>Ссылки на визуализации:</p>
                <ul style="list-style: none; padding: 0;">
                    <li><a href="pca_3d_visualization.html" target="_blank">3D PCA Визуализация (Multi-Parent)</a></li>
                    <li><a href="pca_variance_analysis.html" target="_blank">Анализ дисперсии PCA</a></li>
                    <li><a href="distance_distribution.html" target="_blank">Распределение расстояний</a></li>
                    <li><a href="confidence_entropy_plot.html" target="_blank">Confidence vs Entropy Plot</a></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''

            with open("benchmark_report_multi_parent.html", "w", encoding="utf-8") as f:
                f.write(html_content)

            print("HTML отчет сохранен в benchmark_report_multi_parent.html")

    def run_benchmark(self):
        """Запуск полного бенчмарк-теста"""
        print("=" * 60)
        print("Запуск бенчмарк-теста эмбеддингов (Multi-Parent Mode)")
        print("=" * 60)

        if self.threshold_T is not None:
            print(f"Используется порог T из конфигурации: {self.threshold_T}")
        else:
            print("Порог T будет вычислен как среднее минимальное расстояние")

        # Шаг 1: Загрузка эмбеддингов
        self.load_embeddings()

        # Шаг 2: Выравнивание размерностей
        self.align_dimensions()

        # Шаг 3: Объединение всех эмбеддингов
        all_embeddings = self.combine_all_embeddings()
        print(f"\nОбщее количество эмбеддингов: {all_embeddings.shape}")
        print(f"Из них родительских: {len(self.parent_indices)}")

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

        # Для графика confidence vs entropy создаем отдельные данные
        if test_distances or base_distances:
            viz_metrics = {
                'test_distances': test_distances,
                'base_distances': base_distances
            }
            self.create_confidence_entropy_plot(viz_metrics)

        # Шаг 8: Создание отчета
        self.create_metrics_report(metrics, distances)

        # Шаг 9: Сохранение JSON отчета
        self.save_json_report(metrics, distances)

        # Вывод результатов в консоль
        self.print_results_to_console(metrics, distances)

    def save_json_report(self, metrics, distances):
        """Сохранение полного отчета в JSON формате"""

        # Простая функция для сериализации numpy типов
        def numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_python(item) for item in obj]
            else:
                return obj

        # Подготавливаем метрики
        metrics_clean = metrics.copy()

        # Удаляем временные данные
        metrics_clean.pop('test_distances', None)
        metrics_clean.pop('base_distances', None)

        # Конвертируем numpy типы в метриках
        metrics_clean = numpy_to_python(metrics_clean)

        # Подготавливаем расстояния
        distances_clean = []
        for dist in distances:
            dist_clean = {}
            for key, value in dist.items():
                if key == 'cosine_distances_to_parents' or key == 'cosine_similarities_to_parents':
                    # Конвертируем списки numpy значений
                    dist_clean[key] = [float(v) for v in value]
                elif isinstance(value, (np.integer, np.floating, np.bool_)):
                    dist_clean[key] = numpy_to_python(value)
                else:
                    dist_clean[key] = value
            distances_clean.append(dist_clean)

        # Подготавливаем информацию о родителях
        parent_info_clean = []
        for info in self.parent_info:
            info_clean = {}
            for key, value in info.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    info_clean[key] = numpy_to_python(value)
                else:
                    info_clean[key] = value
            parent_info_clean.append(info_clean)

        report = {
            'metadata': {
                'parent_count': len(self.parent_indices),
                'test_count': len(self.embeddings['test']) if len(self.embeddings['test']) > 0 else 0,
                'base_count': len(self.embeddings['base']) if len(self.embeddings['base']) > 0 else 0,
                'threshold_T': float(metrics['threshold_T']) if metrics['threshold_T'] is not None else None,
                'threshold_source': metrics['threshold_source']
            },
            'metrics': metrics_clean,
            'distances': distances_clean,
            'parent_info': parent_info_clean
        }

        with open("benchmark_full_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("Полный отчет сохранен в benchmark_full_report.json")

    def clean_metrics_for_json(self, metrics):
        """Очистка метрик для JSON сериализации"""
        # Удаляем циклические ссылки и несериализуемые объекты
        if 'test_metrics' in metrics and 'rag_interpretation' in metrics['test_metrics']:
            rag = metrics['test_metrics']['rag_interpretation']
            metrics['test_metrics']['rag_interpretation'] = {
                'interpretation': rag.get('interpretation', ''),
                'impact': rag.get('impact', ''),
                'recommendation': rag.get('recommendation', ''),
                'rag_effect': rag.get('rag_effect', ''),
                'color': rag.get('color', ''),
                'mean_similarity': float(rag.get('mean_similarity', 0)),
                'mean_distance': float(rag.get('mean_distance', 0))
            }

        if 'base_metrics' in metrics and 'rag_interpretation' in metrics['base_metrics']:
            rag = metrics['base_metrics']['rag_interpretation']
            metrics['base_metrics']['rag_interpretation'] = {
                'interpretation': rag.get('interpretation', ''),
                'impact': rag.get('impact', ''),
                'recommendation': rag.get('recommendation', ''),
                'rag_effect': rag.get('rag_effect', ''),
                'color': rag.get('color', ''),
                'mean_similarity': float(rag.get('mean_similarity', 0)),
                'mean_distance': float(rag.get('mean_distance', 0))
            }

        # Удаляем временные данные, которые могут содержать циклические ссылки
        metrics.pop('test_distances', None)
        metrics.pop('base_distances', None)

        # Конвертируем numpy типы в Python типы
        return self.convert_numpy_types(metrics)

    def clean_distances_for_json(self, distances):
        """Очистка расстояний для JSON сериализации"""
        cleaned_distances = []
        for dist in distances:
            cleaned_dist = {
                'index': int(dist['index']),
                'label': str(dist['label']),
                'source': str(dist['source']),
                'min_cosine_distance': float(dist['min_cosine_distance']),
                'max_cosine_distance': float(dist['max_cosine_distance']),
                'mean_cosine_distance': float(dist['mean_cosine_distance']),
                'std_cosine_distance': float(dist['std_cosine_distance']),
                'best_parent': int(dist['best_parent']),
                'best_parent_similarity': float(dist['best_parent_similarity']),
                'parent_confidence': float(dist['parent_confidence']),
                'parent_entropy': float(dist['parent_entropy'])
            }

            # Конвертируем списки расстояний до родителей
            if 'cosine_distances_to_parents' in dist:
                cleaned_dist['cosine_distances_to_parents'] = [
                    float(d) for d in dist['cosine_distances_to_parents']
                ]

            if 'cosine_similarities_to_parents' in dist:
                cleaned_dist['cosine_similarities_to_parents'] = [
                    float(s) for s in dist['cosine_similarities_to_parents']
                ]

            cleaned_distances.append(cleaned_dist)

        return cleaned_distances

    def clean_parent_info_for_json(self):
        """Очистка информации о родителях для JSON"""
        cleaned_info = []
        for info in self.parent_info:
            cleaned_info.append({
                'id': int(info.get('id', 0)),
                'question': str(info.get('question', '')),
                'index': int(info.get('index', 0))
            })
        return cleaned_info

    def convert_numpy_types(self, obj):
        """Рекурсивная конвертация numpy типов в Python типы"""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def print_results_to_console(self, metrics, distances):
        """Вывод результатов в консоль"""
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ БЕНЧМАРК-ТЕСТА (MULTI-PARENT)")
        print("=" * 60)

        print(f"\nКонфигурация:")
        print(f"  Родительских эмбеддингов: {len(self.parent_indices)}")
        print(f"  Исследуемых эмбеддингов: {len(self.embeddings['test']) if len(self.embeddings['test']) > 0 else 0}")
        print(f"  Базовых эмбеддингов: {len(self.embeddings['base']) if len(self.embeddings['base']) > 0 else 0}")
        print(f"  Порог T: {metrics['threshold_T']:.4f} ({metrics['threshold_source']})")

        if metrics.get('multi_parent_metrics'):
            multi = metrics['multi_parent_metrics']
            print(f"\n--- Multi-Parent Метрики ---")
            print(f"Средний Confidence: {multi['mean_confidence']:.4f}")
            print(f"Средняя Entropy: {multi['mean_entropy']:.4f}")
            print(f"Интерпретация: {multi['interpretation']}")

        if metrics.get('parent_coverage'):
            cov = metrics['parent_coverage']
            print(f"\n--- Parent Coverage ---")
            print(f"Покрытие: {cov['coverage_percentage']:.1f}% ({cov['covered_parents']}/{cov['total_parents']})")
            print("Распределение по родителям:")
            for parent_id, count in cov['parent_distribution'].items():
                print(f"  Родитель {parent_id + 1}: {count} чанков")

        if metrics['test_metrics']:
            test_sim = metrics['test_metrics']['mean_cosine_similarity']
            rag_interpretation = metrics['test_metrics']['rag_interpretation']

            print(f"\n--- Исследуемые эмбеддинги ---")
            print(f"Средняя схожесть с лучшим родителем: {test_sim:.4f}")
            print(f"Средний Confidence: {metrics['test_metrics']['mean_parent_confidence']:.4f}")
            print(f"Средняя Entropy: {metrics['test_metrics']['mean_parent_entropy']:.4f}")
            print(f"Стандартное отклонение: {metrics['test_metrics']['std_cosine_similarity']:.4f}")
            print(f"\nИнтерпретация для RAG: {rag_interpretation['interpretation']}")
            print(f"Влияние на RAG: {rag_interpretation['rag_effect']}")

        if metrics['base_metrics']:
            base_sim = metrics['base_metrics']['mean_cosine_similarity']
            rag_interpretation_base = metrics['base_metrics']['rag_interpretation']

            print(f"\n--- Базовые эмбеддинги ---")
            print(f"Средняя схожесть с лучшим родителем: {base_sim:.4f}")
            print(f"Средний Confidence: {metrics['base_metrics']['mean_parent_confidence']:.4f}")
            print(f"Средняя Entropy: {metrics['base_metrics']['mean_parent_entropy']:.4f}")
            print(f"Стандартное отклонение: {metrics['base_metrics']['std_cosine_similarity']:.4f}")
            print(f"\nИнтерпретация для RAG: {rag_interpretation_base['interpretation']}")

        if metrics['comparison']:
            print(f"\n--- Сравнительный анализ ---")
            print(f"Схожесть: {'✓' if metrics['comparison']['test_better_by_similarity'] else '✗'} "
                  f"({metrics['comparison']['similarity_improvement']:+.4f})")
            print(f"Confidence: {'✓' if metrics['comparison']['test_better_by_confidence'] else '✗'} "
                  f"({metrics['comparison']['confidence_improvement']:+.4f})")
            print(f"Entropy: {'✓' if metrics['comparison']['test_better_by_entropy'] else '✗'} "
                  f"({metrics['comparison']['entropy_improvement']:+.4f})")

            print(f"\n--- Итоговые выводы ---")
            print(f"Правило большинства (2 из 3): "
                  f"{'✓ Исследуемый метод лучше' if metrics['comparison']['test_better_by_majority'] else '✗ Базовый метод лучше'}")
            print(f"Композитная оценка: {metrics['comparison']['composite_score']:.4f}")
            print(f"Итоговый вердикт: "
                  f"{'✓ Исследуемый метод лучше' if metrics['comparison']['test_better_by_composite'] else '✗ Базовый метод лучше'}")

        print("\n" + "=" * 60)
        print("Созданы следующие файлы:")
        print("1. pca_3d_visualization.html - 3D визуализация PCA (multi-parent)")
        print("2. pca_variance_analysis.html - Анализ дисперсии")
        print("3. distance_distribution.html - Распределение расстояний")
        print("4. confidence_entropy_plot.html - Confidence vs Entropy plot")
        print("5. benchmark_report_multi_parent.html - Полный HTML отчет")
        print("6. benchmark_full_report.json - Полный JSON отчет")
        print("=" * 60)


# Запуск бенчмарк-теста
if __name__ == "__main__":
    benchmark = EmbeddingBenchmark()
    benchmark.run_benchmark()
