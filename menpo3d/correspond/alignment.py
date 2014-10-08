import pcl
from pcl.registration import icp as pcl_icp
from collections import namedtuple
from menpo.transform import Similarity
import numpy as np

ICPResult = namedtuple('ICPResult', ['didconverge', 'transform'])
as_pcl = lambda p: pcl.PointCloud(p.points.astype(np.float32))


def icp(src, tgt):
    pc_src = as_pcl(src)
    pc_tgt = as_pcl(tgt)
    x = pcl_icp(pc_src, pc_tgt, max_iter=100)
    return ICPResult(x[0], Similarity(x[1].T))
