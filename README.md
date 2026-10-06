# TelaFormer

### Regenerate Icon (`.ico` from `icon.png`):
```powershell
uv run python generate_icon.py
```
*(Or via one-liner)*:
```powershell
uv run python -c "from PIL import Image; Image.open('icon.png').save('icon.ico', format='ICO', sizes=[(16,16), (24,24), (32,32), (48,48), (64,64), (128,128), (256,256)])"
```

### Build Standalone Executable:
```powershell
uv run pyinstaller --noconfirm --onefile --windowed --icon=icon.ico --add-data="icon.png;." --name="TelaFormer" gui.py
```