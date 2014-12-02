from functools import partial
from pathlib import Path
import cPickle as pickle
import gzip

gzip_open = partial(gzip.open, compresslevel=3)

# Lambdas to choose the right settings based on compression on/off
open_for_compress = lambda compress: gzip_open if compress else open
suffixes_for_compress = lambda compress: (['.pickle', '.gz'] if compress
                                          else ['.pickle'])
suffix_for_compress = lambda compress: ''.join(suffixes_for_compress(compress))


def pickle_load(path):
    o = gzip.open if Path(path).suffix == '.gz' else open
    with o(str(path), 'rb') as f:
        x = pickle.load(f)
    if not hasattr(x, 'path'):
        try:
            x.path = Path(path)
        except AttributeError:
            if not 'path' in x:
                x['path'] = Path(path)
    return x


def pickle_dump(x, path, compress=False):
    o, suffixes = open_for_compress(compress), suffixes_for_compress(compress)
    path = Path(path)
    path = (path if path.suffixes == suffixes
            else path.with_suffix(''.join(suffixes)))
    with o(str(path), 'wb') as f:
        pickle.dump(x, f, protocol=2)
