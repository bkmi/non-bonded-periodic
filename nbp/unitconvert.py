import numpy as np

def nondimensionalize():
    pass


def dimensionalize():
    pass


def cart_to_spher(cartesian, *args, **kwargs):
    """convert cartesian coordinates to spherical"""
    r = np.sqrt(np.sum(cartesian**2))
    theta = np.arccos(cartesian[2]/r)
    phi = np.arctan(cartesian[1]/cartesian[0])
    return np.array([r, theta, phi])


def spher_to_cart(spherical, *args, **kwargs):
    """convert spherical to cartesian coordinates"""
    x = spherical[0] * np.sin(spherical[1]) * np.cos(spherical[2])
    y = spherical[0] * np.sin(spherical[1]) * np.sin(spherical[2])
    z = spherical[0] * np.cos(spherical[1])
    return np.array([x, y, z])
