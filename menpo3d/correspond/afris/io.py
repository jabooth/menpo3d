from functools import partial
from menpo3d.io import pickle_load, pickle_dump
from menpo3d.io.pickle import suffix_for_compress


def pickle_dump_with_path_with_id(path_f, x, r, id_, compress=True):
    return pickle_dump(x, path_f(r, id_, compress=compress), compress=compress)


def pickle_load_with_path_with_id(path_f, r, id_, compress=True):
    return pickle_load(path_f(r, id_, compress=compress))


def pickle_dump_with_path(path_f, x, r, compress=True):
    return pickle_dump(x, path_f(r, compress=compress), compress=compress)


def pickle_load_with_path(path_f, r, compress=True):
    return pickle_load(path_f(r, compress=compress))


def generate_load_and_dump(path_f, with_id=True):
    if with_id:
        return (partial(pickle_load_with_path_with_id, path_f),
                partial(pickle_dump_with_path_with_id, path_f))
    else:
        return (partial(pickle_load_with_path, path_f),
                partial(pickle_dump_with_path, path_f))


def path_for_id(dir_f, r, id_, compress=True):
    return dir_f(r) / (id_ + suffix_for_compress(compress))

generate_path = lambda dir_f: partial(path_for_id, dir_f)

shape_dir = lambda r: r / 'shape'
texture_dir = lambda r: r / 'texture'
metadata_dir = lambda r: r / 'metadata'

afr_path = lambda r, compress=False: r / ('afr' +
                                          suffix_for_compress(compress))
shape_path = generate_path(shape_dir)
texture_path = generate_path(texture_dir)
metadata_path = generate_path(metadata_dir)

load_afr, dump_afr = generate_load_and_dump(afr_path, with_id=False)
load_shape, dump_shape = generate_load_and_dump(shape_path)
load_texture, dump_texture = generate_load_and_dump(texture_path)
load_metadata, dump_metadata = generate_load_and_dump(metadata_path)


def prepare_afr(r):
    if not r.is_dir():
        r.mkdir()
    shape = shape_dir(r)
    if not shape.is_dir():
        shape.mkdir()
    texture = texture_dir(r)
    if not texture.is_dir():
        texture.mkdir()
    metadata = metadata_dir(r)
    if not metadata.is_dir():
        metadata.mkdir()


def dump_lafr_result(x, r, id_, compress=True):
    dump_metadata({'sparse_3d': x.sparse_3d}, r, id_, compress=compress)
    dump_shape(x.shape_image, r, id_, compress=compress)
    dump_texture(x.texture_image, r, id_, compress=compress)
