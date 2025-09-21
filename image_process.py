from PIL import Image, ImageEnhance


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def image_enhance(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    return image
    
def resize_image(image: Image.Image, size=(640, 480)) -> Image.Image:
    if image.width <= size[0] and image.height <= size[1]:
        return image
    else:
        image.thumbnail(size, Image.LANCZOS)
        new_img = Image.new("RGB", size, (255, 255, 255))
        left = (size[0] - image.width) // 2
        top = (size[1] - image.height) // 2
        new_img.paste(image, (left, top))
        return new_img
