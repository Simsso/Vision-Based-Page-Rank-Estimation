from skimage import io


def img_loading_possible(img_path: str) -> bool:
    try:
        io.imread(img_path)
        return True
    except ValueError:
        return False

