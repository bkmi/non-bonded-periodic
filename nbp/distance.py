import numpy as np
import nbp


class Distance:
    def __init__(self, system):
        self._distance_vectors = None
        self._distances_unwrapped = None
        self._distances_wrapped = None
        self._system = system

    def system(self):
        return self._system

    def positions(self):
        return self.system().state().positions()

    def distance_vectors_unwrapped(self):
        if self._distance_vectors is None:
            unwrapped = self.positions()[None, :, :] - self.positions()[:, None, :]
            self._distance_vectors = unwrapped
        return self._distance_vectors

    def distance_vectors_wrapped(self):
        if self._distance_vectors is None:
            unwrapped = self.distance_vectors_unwrapped()
            wrapped = np.apply_along_axis(lambda x: nbp.periodic_wrap_corner(x, self._system.info().char_length()),
                                          -1, unwrapped)
            self._distance_vectors = wrapped
        return self._distance_vectors

    def distances_unwrapped(self):
        if self._distances_unwrapped is None:
            self._distances_unwrapped = np.linalg.norm(self.distance_vectors_unwrapped(), axis=-1)
        return self._distances_unwrapped

    def distances_wrapped(self):
        if self._distances_wrapped is None:
            self._distances_wrapped = np.linalg.norm(self.distance_vectors_wrapped(), axis=-1)
        return self._distances_wrapped