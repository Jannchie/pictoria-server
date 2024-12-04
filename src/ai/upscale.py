import PIL.Image
from image_gen_aux import UpscaleWithModel


def upscale_image(image: PIL.Image.Image):
    upscaler = UpscaleWithModel.from_pretrained("OzzyGT/DAT_X2").to("cuda")
    return upscaler(image, tiling=True, tile_width=1024, tile_height=1024)  # type: ignore


if __name__ == "__main__":
    upscale_image(PIL.Image.open(R"C:\Users\Jannchie\Downloads\BnwX8prwWZKb8.jpg")).show()
