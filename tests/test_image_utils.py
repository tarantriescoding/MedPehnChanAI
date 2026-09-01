import numpy as np

from image_utils import detect_edges, preprocess_image


def test_preprocess_image_converts_to_grayscale():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    processed = preprocess_image(image)
    assert processed.shape == (50, 50)
    assert processed.dtype == np.uint8


def test_detect_edges_returns_image():
    image = np.zeros((50, 50), dtype=np.uint8)
    edges = detect_edges(image)
    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
