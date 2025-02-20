from pathlib import Path
PRJ_ROOT = Path(__file__).parents[1]
SRC_ROOT = PRJ_ROOT / 'lcs'

import logging
import logging.config

logging.config.fileConfig(SRC_ROOT / 'logging.conf')

# document dimensions
INCH = 1
PAGEWIDTH = 8.5*INCH
PAGEHEIGHT = 11*INCH
TEXTWIDTH = 5.5 * INCH
FIGURE_PATH = PRJ_ROOT / 'tex_neurips24/fig'

from lcs.init_mpl import *  # runs the plotting setup
import re