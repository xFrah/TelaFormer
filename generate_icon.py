"""Helper script to regenerate icon.ico from icon.png with standard Windows resolutions."""
from pathlib import Path
from PIL import Image

def generate_ico(png_path="icon.png", ico_path="icon.ico"):
    src = Path(png_path)
    if not src.exists():
        print(f"Error: '{png_path}' not found.")
        return False

    img = Image.open(src)
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, format="ICO", sizes=sizes)
    print(f"Successfully generated '{ico_path}' with {len(sizes)} mipmap resolutions from '{png_path}'.")
    return True

if __name__ == "__main__":
    generate_ico()
