"""Central application state for the Single-Cell Explorer GUI.
 
All panels observe this state reactively via param watchers.
"""
 
import param
import pandas as pd
 
 
class AppState(param.Parameterized):
    """Shared state container for the SCExplorer application."""
 
    # Database summary (loaded once on init)
    all_experiments_df = param.DataFrame(
        default=pd.DataFrame(columns=['exp_name', 'n_cells', 'cell_types', 'protocols_ran', 'species']),
        doc="Summary of all single-cell experiments from get_single_cell_dataset_summary()"
    )
    all_species = param.List(default=[], doc="Unique species in the database")
 
    # Species pre-filter (gates which experiments are available to add)
    selected_species = param.List(default=[], doc="Species currently selected in pre-filter")
 
    # Loaded experiments
    loaded_exp_names = param.List(default=[], doc="exp_names user has added")
    exp_summaries = param.Dict(default={}, doc="{exp_name: DataFrame from get_exp_summary()}")
 
    # Tree filters (applied to data tree, not experiment selection)
    protocol_filter = param.String(default='', doc="Protocol name filter text")
    protocol_match_mode = param.Selector(
        objects=['contains', 'equals'], default='contains',
        doc="How to match protocol filter"
    )
    celltype_filter = param.List(default=[], doc="Cell types to show")
    recording_technique_filter = param.List(default=[], doc="Recording techniques to show")
    custom_filters = param.List(default=[], doc="List of (column, operator, value) tuples")
 
    # Selection (immediate plot trigger)
    selected_epochs = param.List(
        default=[],
        doc="List of (exp_name, block_id, epoch_idx) tuples currently selected"
    )
 
    # Loaded data cache (lazy — populated on tree expand / epoch select)
    loaded_blocks = param.Dict(
        default={},
        doc="{(exp_name, block_id): (StimBlock, SCResponseBlock)}"
    )
 
    # Analysis
    active_analyses = param.List(default=[], doc="List of active AnalysisPlugin instances")
 
    def initialize(self):
        """Load the experiment summary from the database."""
        from retinanalysis.utils.datajoint_utils import get_single_cell_dataset_summary
        self.all_experiments_df = get_single_cell_dataset_summary(verbose=False)
        species_list = sorted(self.all_experiments_df['species'].dropna().unique().tolist())
        self.all_species = species_list
        self.selected_species = list(species_list)
 
    @property
    def available_experiments(self):
        """Experiments matching the current species pre-filter."""
        df = self.all_experiments_df
        if not self.selected_species:
            return df
        return df[df['species'].isin(self.selected_species)]
 
    def add_experiment(self, exp_name):
        """Load an experiment and add it to the session."""
        if exp_name in self.loaded_exp_names:
            return
        try:
            from retinanalysis.utils.datajoint_utils import get_exp_summary
            df_exp = get_exp_summary(exp_name)
            if df_exp is None or len(df_exp) == 0:
                print(f"[SCExplorer] No data found for experiment '{exp_name}'.")
                return
            summaries = dict(self.exp_summaries)
            summaries[exp_name] = df_exp
            self.exp_summaries = summaries
            self.loaded_exp_names = self.loaded_exp_names + [exp_name]
        except Exception as e:
            print(f"[SCExplorer] Error loading experiment '{exp_name}': {e}")
 
    def remove_experiment(self, exp_name):
        """Unload an experiment from the session."""
        if exp_name not in self.loaded_exp_names:
            return
        summaries = dict(self.exp_summaries)
        summaries.pop(exp_name, None)
        self.exp_summaries = summaries
        self.loaded_exp_names = [n for n in self.loaded_exp_names if n != exp_name]
        # Remove associated selections
        self.selected_epochs = [
            s for s in self.selected_epochs if s[0] != exp_name
        ]
        # Remove cached blocks
        blocks = dict(self.loaded_blocks)
        for key in list(blocks.keys()):
            if key[0] == exp_name:
                del blocks[key]
        self.loaded_blocks = blocks
 
    def select_block(self, exp_name, block_id):
        """Load a block and select all its epochs for plotting."""
        try:
            sb, rb = self.get_or_load_block(exp_name, block_id, b_spiking=True)
            n_epochs = len(rb.amp_data)
            epochs = [(exp_name, block_id, i) for i in range(n_epochs)]
            self.selected_epochs = epochs
            return sb, rb
        except Exception as e:
            print(f"[SCExplorer] Error loading block {block_id}: {e}")
            return None

    def get_or_load_block(self, exp_name, block_id, b_spiking=True):
        """Get cached StimBlock/SCResponseBlock or load them."""
        key = (exp_name, block_id)
        if key not in self.loaded_blocks:
            from retinanalysis.classes.stim import StimBlock
            from retinanalysis.classes.response import SCResponseBlock
            sb = StimBlock(exp_name, block_id, verbose=False)
            rb = SCResponseBlock(exp_name, block_id, b_spiking=b_spiking, verbose=False)
            blocks = dict(self.loaded_blocks)
            blocks[key] = (sb, rb)
            self.loaded_blocks = blocks
        return self.loaded_blocks[key]