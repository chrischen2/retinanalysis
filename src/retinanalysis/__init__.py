# Must be the first import, otherwise the database won't load properly
import retinanalysis.config.schema as schema

# Import various data and analysis directories directly.
# Settings doesn't reference any of the utils or classes so it should be
# safe to import first without circular import issues
from . import config
from .config import settings
from .config.settings import (ANALYSIS_DIR,
                              DATA_DIR,
                              RAW_DIR,
                              H5_DIR,
                              META_DIR,
                              TAGS_DIR,
                              QUERY_DIR,
                              OUTPUT_DIR,
                              USER,
                              PROTOCOL_REPOS_ROOT,
                              find_protocol_repo)

# Utilities imported first. They should NEVER reference the classes for anything
# other than type hints, which should be done using the TYPE_CHECKING and 
# __future__ annotations imports (see vision_utils). This avoids problems with
# circular imports
from . import utils
from .utils import database_pop
from .utils.database_pop import *

from .utils import database_utils
from .utils.database_utils import *

from .utils import datajoint_utils
from .utils.datajoint_utils import *

from .utils import ei_utils
from .utils.ei_utils import *

from .utils import regen
from .utils.regen import *

from .utils import vision_utils
from .utils.vision_utils import *

from .utils import cell_type_utils
from .utils.cell_type_utils import (load_canonical_cell_types,
                                    map_cell_type,
                                    filter_available_types)

from .utils import mosaic_overlay
from .utils.mosaic_overlay import (plot_stim_with_mosaic, cells_inside_array,
                                   electrode_positions_canvas_px,
                                   electrode_geometry)

from .utils import psth as _psth_mod
from .utils.psth import (spike_times_to_psth,
                         epoch_spikes_to_psth,
                         psth_time_axis,
                         gaussian_filter_1d)

from .utils import raster as _raster_mod
from .utils.raster import plot_raster_with_psth

from .utils import cell_type_check
from .utils.cell_type_check import (plot_cell_type_check,
                                    plot_cell_type_grid)

from .utils import visual_qc as _visual_qc_mod
from .utils.visual_qc import (browse_cells_qc, load_visual_qc,
                              visual_qc_csv_path, select_good_cells)

from .utils import cell_match as _cell_match_mod
from .utils.cell_match import (compute_ei_stats, build_cell_match_table,
                               save_cell_match, load_cell_match,
                               cell_match_csv_path)

from .utils import protocol_qc as _protocol_qc_mod
from .utils.protocol_qc import (save_protocol_qc, load_protocol_qc,
                                protocol_qc_csv_path)

from .utils import offline_store as _offline_store_mod
from .utils.offline_store import (OfflineDataset,
                                  save_offline_data,
                                  load_offline_data,
                                  load_or_build_offline,
                                  load_offline_many,
                                  offline_h5_path)

# Raw-trace sorting QC. Imported lazily inside its own module to keep
# bin2py off the critical-path import (it's only needed when this code is
# actually invoked).
from .classes.raw import (RawTraces,
                          plot_sts_over_trace,
                          plot_sorting_qc,
                          plot_sorting_qc_grid,
                          primary_electrode_of_cell,
                          cells_sharing_electrode)

from .utils import sorting_qc as _sorting_qc_mod
from .utils.sorting_qc import (sample_sorting_qc_cells,
                               sample_and_plot_sorting_qc,
                               sorting_qc_gui)

from .utils import local_cache as _local_cache_mod
from .utils.local_cache import (mirror_to_local_cache,
                                 compare_cache_vs_source,
                                 local_cache_status,
                                 clear_local_cache,
                                 create_mea_pipeline_cached,
                                 load_cached_pipeline,
                                 pipeline_cache_path,
                                 LOCAL_CACHE_ROOT)

# Session-wide network bandwidth gauge (counter + widget). The counter
# is fed by find_path() in config.settings; widget is in utils.bandwidth.
from .config.settings import (network_bytes_resolved,
                                network_resolutions_count,
                                reset_network_gauge)
from .utils.bandwidth import (print_network_gauge,
                                network_bandwidth_gauge_widget,
                                bandwidth_scope)

# Per-protocol analyzers live under retinanalysis.protocols; import the
# package but don't pull every module at top level (loaded on demand).
from . import protocols

# `parse_data` pulls scipy.signal (~1s). It isn't used at module level — load it
# lazily via `from retinanalysis.utils import parse_data` when actually needed.

# Import classes last
from . import classes
from .classes import analysis_chunk
from .classes.analysis_chunk import AnalysisChunk

from .classes import stim
from .classes.stim import (StimBlock,
                           MEAStimBlock,
                           MEAStimGroup,
                           create_mea_stim_group,
                           D_REGEN_FXNS)

from .classes import response
from .classes.response import (ResponseBlock,
                               MEAResponseBlock,
                               SCResponseBlock,
                               MEAResponseGroup,
                               check_frame_times,
                               create_mea_response_group)

from .classes import qc
from .classes.qc import MEAQC


# Stimulus regeneration (depends on classes for type hints only). Import after
# classes so registry side-effects can reference StimBlock-style objects.
from . import regen as _regen_pkg
from .regen import (regen_stimulus,
                    available_protocols,
                    render_displayed_canvas)

# Pipeline must be imported last as it references the above pieces.
from .classes import mea_pipeline
from .classes.mea_pipeline import (MEAPipeline,
                                   create_mea_pipeline)
from .classes import sc_pipeline
from .classes import dedup
from .classes.dedup import DedupBlock

# GUI (lazy import — only loaded when accessed)
from . import DJ_GUI

# Standalone single-cell analysis utilities
from . import SCutils

# End-to-end experiment-archive driver (build pipeline + QC + per-cell PNGs).
from . import analyze
from .analyze import (analyze_experiment, analyze_experiments,
                       detect_ss_version, SS_VERSION_PRIORITY)
 



