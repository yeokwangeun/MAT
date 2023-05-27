# vim: expandtab:ts=4:sw=4
import numpy as np

class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, z, w, h, d)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, top left z, width, height, depth)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, min z, max x, max y, max z)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[3:] += ret[:3]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, center z, x/y ratio, y/z ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:3] += ret[3:] / 2
        ret[3] /= ret[4]
        ret[4] /= ret[5]
        return ret
