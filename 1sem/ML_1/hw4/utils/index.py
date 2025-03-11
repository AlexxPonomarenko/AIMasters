# INDEX.PY

from time import time as tm
import faiss
import hnswlib


def timer(func):
    '''
    декоратор, замеряющий время работы функции
    '''
    def wrapper(*args, **kwargs):
        start_time = tm()
        result = func(*args, **kwargs)
        end_time = tm() - start_time
        if isinstance(result, tuple):
            return *result, end_time
        return result, end_time
    return wrapper


@timer
def build_IVFPQ(build_data, **fixed_params):
    dim = fixed_params['dim']
    coarse_index = fixed_params['coarse_index']
    nlist = fixed_params['nlist']
    m = fixed_params['m']
    nbits = fixed_params['nbits']
    metric = fixed_params['metric']
    
    num_threads = fixed_params.get('num_threads', 1)
    faiss.omp_set_num_threads(num_threads)
    
    index = faiss.IndexIVFPQ( # у faiss туго с именованными аргументами
        coarse_index, # индекс для поиска соседей-центроидов
        dim, # размерность исходных векторов
        nlist, # количество coarse-центроидов = ячеек таблицы
        m, # на какое кол-во подвекторов бить исходные для PQ
        nbits, # log2 k* - количество бит на один маленький (составной) PQ-центроид
        metric # метрика, по которой считается расстояние между остатком(q) и [pq-центроидом остатка](x)
    )
    index.train(build_data)
    index.add(build_data)
    return index # из-за декоратора ожидайте, что возвращается index, build_time

@timer
def search_faiss(index, query_data, k, nprobe=1):
    index.nprobe = nprobe # количество ячеек таблицы, в которые мы заглядываем. Мы заглядываем в nprobe ближайших coarse-центроидов для q
    distances, labels = index.search(query_data, k)
    return distances, labels # из-за декоратора ожидайте, что возвращается distances, labels, search_time

@timer
def build_hnsw(build_data, **fixed_params):
    dim = fixed_params['dim']
    space = fixed_params['space']
    M = fixed_params['M']
    ef_construction = fixed_params['ef_construction']

    # Declaring index
    index = hnswlib.Index(space = space, dim = dim)

    # Initializing index - the maximum number of elements should be known beforehand
    num_elements = build_data.shape[0]
    index.init_index(max_elements = num_elements, ef_construction = ef_construction, M = M)

    # Element insertion (can be called several times):
    index.add_items(build_data)

    return index

@timer
def search_hnsw(index, query_data, k_neighbors, efSearch):
    # Controlling the recall by setting ef:
    index.set_ef(efSearch)

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = index.knn_query(query_data, k_neighbors)

    return distances, labels

@timer
def build_IVFFlat(build_data, **fixed_params):
    dim = fixed_params['dim']
    coarse_index = fixed_params['coarse_index']
    nlist = fixed_params['nlist']
    metric = fixed_params['metric']

    num_threads = fixed_params.get('num_threads', 1)
    faiss.omp_set_num_threads(num_threads)

    index = faiss.IndexIVFFlat(coarse_index, dim, nlist, metric)
    index.train(build_data)
    index.add(build_data)

    return index

@timer
def build_flat_l2(build_data, dim):
    index = faiss.IndexFlatL2(dim)
    index.train(build_data)
    index.add(build_data)
    return index

@timer
def build_flat_ip(build_data, dim):
    index = faiss.IndexFlatIP(dim)
    index.train(build_data)
    index.add(build_data)
    return index

@timer
def search_flat(index, query_data, k):
    distances, labels = index.search(query_data, k)
    return distances, labels