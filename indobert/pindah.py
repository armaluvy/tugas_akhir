import os
import tempfile
import os

# Set folder temp
tempfile.tempdir = 'E:/temp_python'

# Jika perlu, pastikan foldernya dibuat dulu
os.makedirs('E:/temp_python', exist_ok=True)
# Ganti ke direktori selain C
os.environ['TRANSFORMERS_CACHE'] = 'E:/huggingface_cache'

from transformers.utils import logging
logging.set_verbosity_info()