# cistemx.io

File I/O utilities for cryo-EM data formats.

## Modules

### mrc.py

MRC image file operations.

```python
from cistemx.io.mrc import load_mrc_image, get_mrc_pixel_size

# Load image and pixel size
image, pixel_size = load_mrc_image('/path/to/mip.mrc')

# Just get pixel size (faster, no data read)
pixel_size = get_mrc_pixel_size('/path/to/mip.mrc')
```

## TODO

- Add STAR file utilities if needed (currently not used by our scripts)
- Consider adding MRC write functionality
