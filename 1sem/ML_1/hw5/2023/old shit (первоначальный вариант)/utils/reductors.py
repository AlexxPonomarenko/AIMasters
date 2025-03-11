from umap import UMAP
import openTSNE
from sklearn.decomposition import PCA
from time import time as tm

# при желании можете использовать время работы в рисовалке.
# сейчас оно не используется

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = tm()
        result = func(*args, **kwargs)
        end_time = tm() - start_time
        if isinstance(result, tuple):
            return *result, end_time
        return result, end_time
    return wrapper


# при желании, UMAP можно изменить, чтобы он тоже принимал предпосчитанные affinities, init
@timer
def make_umap(data, params, y=None):
    '''
    можно вшить y через partial для [semi-]supervised learning
    '''
    embedding = UMAP(**params).fit_transform(data, y)
    return embedding


@timer
def make_tsne(data, params, init=None, affinities=None):
    '''
    можно вшить init, affinities через partial, чтобы не считать по сто раз,
        если вы не хотите их менять
    '''
    rescaled_init = None
    if init is not None:
        rescaled_init = openTSNE.initialization.rescale(init, inplace=False, target_std=0.0001)
        
    embedding = openTSNE.TSNE(**params).fit(data, initialization=rescaled_init, affinities=affinities)
    return embedding

@timer
def make_pca(data, params):
    embedding = PCA(**params).fit_transform(data)
    return embedding