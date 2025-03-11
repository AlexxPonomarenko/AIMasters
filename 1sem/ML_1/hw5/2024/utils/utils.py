import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class KNNFeatureAggregator:
    def __init__(self, index_info):
        self.index_info = index_info

    def train(self, train_data, index_add_info=None):
        train_data_np = train_data.to_numpy().astype(np.float32)

        fixed_params = self.index_info['fixed_params']
        build_func = self.index_info['build_func']

        self.index, build_time = build_func(train_data_np, **fixed_params)
        print(f"Index is finally built. Total index building time : {build_time} s")

    def kneighbors(self, query_data, index_add_info, k=100, is_train=True):
        query_data_np = query_data.to_numpy().astype(np.float32)

        search_param = index_add_info['search_param'][1]
        search_func = index_add_info['search_func']
        if is_train:
            _, labels, _ = search_func(self.index, query_data_np, k+1, search_param)
            mask = labels != np.arange(labels.shape[0])[:, np.newaxis]
            labels = labels[mask].reshape(query_data.shape[0], k)
        else:
            _, labels, _ = search_func(self.index, query_data_np, k, search_param)

        return labels

    def make_features(self, neighbor_ids, train_data, feature_info):
        new_features = pd.DataFrame()

        for new_col_name, params in feature_info.items():
            orig_col_name, agg_func, nn = params
            if isinstance(nn, int):
                nn_ids = neighbor_ids[:,:nn].astype(np.int64)
                nn_data = np.take(train_data.loc[:, orig_col_name].to_numpy(), nn_ids)
                new_column = np.apply_along_axis(agg_func, axis=1, arr=nn_data)
                new_features[new_col_name] = new_column
            elif isinstance(nn, list):
                for n in nn:
                    new_name = new_col_name + "_" + str(n) + "nn"
                    nn_ids = neighbor_ids[:,:n].astype(np.int64)
                    nn_data = np.take(train_data.loc[:, orig_col_name].to_numpy(), nn_ids)
                    new_column = np.apply_along_axis(agg_func, axis=1, arr=nn_data)
                    new_features[new_name] = new_column
            else:
                print("Third parameter must be a single integer or a list!")

        return new_features

def calc_recall(true_labels, pred_labels, k, exclude_self=False, return_mistakes=False):
    '''
    счиатет recall@k для приближенного поиска соседей
    
    true_labels: np.array (n_samples, k)
    pred_labels: np.array (n_samples, k)
    
    exclude_self: bool
        Если query_data была в трейне, считаем recall по k ближайшим соседям, не считая самого себя
    return_mistakes: bool
        Возвращать ли ошибки
    
    returns:
        recall@k
        mistakes: np.array (n_samples, ) с количеством ошибок
    '''
    n = true_labels.shape[0]
    n_success = []
    shift = int(exclude_self)
    
    for i in range(n):
        n_success.append(np.intersect1d(true_labels[i, shift:k+shift], pred_labels[i, shift:k+shift]).shape[0])
        
    recall = sum(n_success) / n / k
    if return_mistakes:
        mistakes = k - np.array(n_success)
        return recall, mistakes
    return recall


def plot_ann_performance(
    build_data,
    query_data,
    index_dict,
    k,
    flat_build_func,
    flat_search_func,
    query_in_train : bool,
    qps_line = None,
    recall_line = None,
    title = None,
):
    '''
    build_data: data, на которой будут строиться индексы подаваемых на вход алгоритмов
    query_data: data, для которой будут искаться соседи
    index_dict: словарик {'index_name': словарик с необходимым*, ...} с необходимой инфой для каждого из исследуемых алгоритмов aNN
    k: для меры качества
    flat_build_func: функция, которая строит Flat-индекс
    flat_search_func: функция, которая ищет в Flat-индексе
    query_in_train: флаг того, что query_data содержится в build_data. Если это так, мерим качество по k соседям без учета ближайшего
    qps_line: float. Если указано, нарисуем горизонтальную линию по этому значению
    recall_line: float. Если указано, нарисуем вертикальную линию по этому значению
    title: str. Если указан, сделать у графика такой title
    '''

    # формируем макет будущего графика
    fig, axes = plt.subplots(1, 2, figsize = (15,8))

    # воспользуемся обычными (flat) функциями  (посчитаем идеальный вариант)
    flat_index, flat_build_time = flat_build_func(build_data)
    flat_distances, flat_labels, flat_search_time = flat_search_func(flat_index, query_data, k)

    # посчитаем скорость flat поиска (измеряется в queries / sec)
    flat_speed = len(query_data) / flat_search_time

    # изобразим на графике эту скорость
    axes[1].axhline(y = flat_speed, color = 'darkgreen', linestyle = '--', label = f'Flat speed = {round(flat_speed, 1)} qps')

    # теперь пройдемся по всем алгоритмам, переданным через index_dict
    index_names = []
    build_times = []
    speeds = []
    recalls = []

    for index_name, info in index_dict.items():

        # из каждого алгоритма получим его параметры
        fixed_params = info['fixed_params']
        build_func = info['build_func']
        search_param = info['search_param']
        search_func = info['search_func']

        # сформируем индексы и посчитаем времена на создание
        index, build_time = build_func(build_data, **fixed_params)
        index_names.append(index_name)
        build_times.append(build_time)

        # разобьем словарь search_param на название и список значений
        search_param_name = search_param[0]
        search_param_values = search_param[1]

        for search_param_value in search_param_values:
            # посчитаем скорости поиска
            distances, labels, search_time = search_func(index, query_data, k, search_param_value)
            speed = len(query_data) / search_time
            speeds.append(speed)

            # посчитаем точности поиска
            recall = calc_recall(flat_labels, labels, k - int(query_in_train), query_in_train)
            recalls.append(recall)

            # ставим точки на графике и подписываем их
            axes[1].plot(recall, speed, 'o')
            axes[1].annotate(f'{search_param_name} = {search_param_value}', (recall, speed), ha = 'center', xytext=(0,10), textcoords = 'offset points')
        
        # соединям полученные точки
        axes[1].plot(recalls[-len(search_param_values):], speeds[-len(search_param_values):], '-', label = index_name)
    
    # делаем темную сетку
    sns.set(style="darkgrid")

    # форма первого графика
    sns.barplot(x=index_names, y=build_times, palette='dark', ax=axes[0])
    
    # устанавливаем название для первого графика и его осей координат
    axes[0].set_title('BUILD_TIME')
    """ Это не получилось :( Хотел сделать вертикальные повернутые на 90 градусов надписи под столбцами
    for i, p in enumerate(axes[0].patches):
        axes[0].annotate(f'{index_names[i]}', xy = (0,0.2), ha = 'center', va = 'bottom', rotation = 90, color = 'black', textcoords = 'offset points')"""
    axes[0].set_ylabel('Time (s)')


    # если есть название для второго графика - устанавливаем его
    if title:
        axes[1].set_title(title)

    # подписываем оси
    axes[1].set_xlabel(f'Recall@{k}')
    axes[1].set_ylabel('Queries per second (qps)')
    axes[1].set_yscale('log')

    # рисуем вертикальную и горизовнтальную прямые, если нужно
    if qps_line:
        axes[1].axhline(y = qps_line, color = 'red', linestyle = '--', label = f'qps_line = {round(qps_line,2)} qps')
    if recall_line:
        axes[1].axvline(x = recall_line, color = 'purple', linestyle = '--', label = f'recall_line = {round(recall_line,2)}')

    # помещаем легенду в правый верхний угол
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

    
def analyze_ann_method(
    build_data,
    query_data,
    build_func,
    search_func,
    k,
    flat_build_func,
    flat_search_func,
    query_in_train : bool,
    index_name = None
):
    '''
    build_data: data, на которой будут строиться индексы подаваемых на вход алгоритмов
    query_data: data, для которой будут искаться соседи
    build_func: функция для построения индекса
    search_func: функция для поиска в индексе
    k: для меры качества
    flat_build_func: функция, которая строит Flat-индекс
    flat_search_func: функция, которая ищет в Flat-индексе
    query_in_train: флаг того, что query_data содержится в build_data. Если это так, мерим качество по k соседям без учета ближайшего
    index_name: str. Если указан, сделать у графика такой index_name
    '''
    # Воспользуемся обычными (flat) функциями (посчитаем идеальный вариант)
    flat_index, flat_build_time = flat_build_func(build_data)
    flat_distances, flat_labels, flat_search_time = flat_search_func(flat_index, query_data, k)

    # Поищем соседей с помощью особых функций
    index, build_time = build_func(build_data)
    distances, labels, search_time = search_func(index, query_data, k)

    # Посчитаем точность поиска
    recall, mistakes = calc_recall(flat_labels, labels, k - int(query_in_train), query_in_train, True)

    # Подсчет количества ошибок для каждого значения
    error_counts = np.bincount(mistakes)

    # Добавляем в конец массива k - len(error_counts) + 1 нулевых элементов чтобы была информация о последних соседях
    error_counts = np.pad(error_counts, (0, k - len(error_counts) + 1), constant_values = 0)

    # У всех элементов, у которых кол-во ошибок 0, заменяем на -10
    error_counts = np.where(error_counts == 0, -10, error_counts)

    # Создание графика
    bars = plt.bar(range(len(error_counts)), error_counts, tick_label=range(len(error_counts)), bottom = 0, 
                   label = f'build time: {round(build_time,2)}, sec \nqps: {round(len(query_data) / build_time, 2)} \nrecall@{k}: {round(recall, 2)}')
    plt.ylim(min(error_counts) - 10, max(error_counts) + 20)

    # Добавим название графика, если нужно
    if index_name:
        plt.title(index_name)
    
    # Добавление подписей
    plt.xlabel('Количество ошибок')
    plt.ylabel('Количество элементов')

    # Подпись каждого столбца сверху значением y
    for bar, value in zip(bars, error_counts):
        display_value = max(0, value) # Если значение отрицательное то используем 0
        plt.text(bar.get_x() + bar.get_width() / 2, display_value, str(display_value), ha='center', va='bottom')

    # Отображение легенды
    plt.legend()

    # Отображение графика
    plt.show()



# Для FASHION MNIST
def knn_predict_classification(neighbor_ids, tr_labels, n_classes, distances=None, weights='uniform'):
    '''
    по расстояниям и айдишникам получает ответ для задачи классификации
    
    distances: (n_samples, k) - расстояния до соседей
    neighbor_ids: (n_samples, k) - айдишники соседей
    tr_labels: (n_samples,) - метки трейна
    n_classes: кол-во классов
    
    returns:
        labels: (n_samples,) - предсказанные метки
    '''
    
    n, k = neighbor_ids.shape

    labels = np.take(tr_labels, neighbor_ids)
    labels = np.add(labels, np.arange(n).reshape(-1, 1) * n_classes, out=labels)

    if weights == 'uniform':
        w = np.ones(n * k)
    elif weights == 'distance' and distances is not None:
        w = 1. / (distances.ravel() + 1e-10)
    else:
        raise NotImplementedError()
        
    labels = np.bincount(labels.ravel(), weights=w, minlength=n * n_classes)
    labels = labels.reshape(n, n_classes).argmax(axis=1).ravel()
    return labels


# Для крабов!
def get_k_neighbors(distances, k):
    '''
    считает по матрице попарных расстояний метки k ближайших соседей
    
    distances: (n_queries, n_samples)
    k: кол-во соседей
    
    returns:
        labels: (n_queries, k) - метки соседей
    '''
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    lowest_distances = np.take_along_axis(distances, indices, axis=1)
    neighbors_idx = lowest_distances.argsort(axis=1)
    indices = np.take_along_axis(indices, neighbors_idx, axis=1) # sorted
    sorted_distances = np.take_along_axis(distances, indices, axis=1)
    return sorted_distances, indices
