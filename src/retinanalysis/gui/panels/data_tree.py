"""Hierarchical data tree browser.

Displays: exp_name → Cell → EpochGroup → EpochBlock → Epoch
with checkboxes for selection at every level.
"""

import panel as pn
import param

from retinanalysis.gui.state import AppState


class _TreeNode:
    """Internal representation of a tree node."""

    __slots__ = ('node_id', 'label', 'level', 'data', 'children', 'parent_id')

    def __init__(self, node_id, label, level, data=None, parent_id=None):
        self.node_id = node_id
        self.label = label
        self.level = level  # 'experiment', 'cell', 'epoch_group', 'epoch_block', 'epoch'
        self.data = data or {}
        self.children = []
        self.parent_id = parent_id


class DataTree(pn.viewable.Viewer):
    """Hierarchical tree browser built from loaded experiment summaries.

    The tree is rendered as HTML with checkboxes.  JS callbacks handle
    parent/child propagation and send selections back to Python via a
    hidden TextInput bridge.
    """

    state = param.ClassSelector(class_=AppState)

    def __init__(self, state, **params):
        super().__init__(state=state, **params)

        self._html_pane = pn.pane.HTML(
            "", sizing_mode='stretch_width', min_height=200,
        )
        # Bridge widget: JS writes JSON selection here, Python watches it
        self._bridge_id = f"sc_tree_bridge_{id(self)}"
        self._selection_bridge = pn.widgets.TextInput(
            value='[]', visible=False,
            css_classes=[self._bridge_id],
        )
        self._selection_bridge.param.watch(self._on_selection_bridge, 'value')

        # Epoch cache: {(exp_name, block_id): df_epochs}
        self._epoch_cache = {}

        # Rebuild tree when experiments or filters change
        for p in ['loaded_exp_names', 'exp_summaries', 'protocol_filter',
                   'protocol_match_mode', 'celltype_filter',
                   'recording_technique_filter', 'custom_filters']:
            state.param.watch(self._rebuild, p)

        self._rebuild()

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build_tree_data(self):
        """Build the nested tree structure from loaded experiment summaries."""
        roots = []
        for exp_name in self.state.loaded_exp_names:
            df = self.state.exp_summaries.get(exp_name)
            if df is None or df.empty:
                continue

            # Look up species
            row_all = self.state.all_experiments_df[
                self.state.all_experiments_df['exp_name'] == exp_name
            ]
            species = row_all['species'].values[0] if len(row_all) > 0 else ''

            exp_node = _TreeNode(
                node_id=f"exp:{exp_name}",
                label=f"{exp_name} ({species})" if species else exp_name,
                level='experiment',
                data={'exp_name': exp_name},
            )

            # Group by cell_label
            if 'cell_label' not in df.columns:
                roots.append(exp_node)
                continue

            for cell_label, df_cell in df.groupby('cell_label', sort=True):
                cell_node = _TreeNode(
                    node_id=f"cell:{exp_name}:{cell_label}",
                    label=str(cell_label),
                    level='cell',
                    data={'exp_name': exp_name, 'cell_label': cell_label},
                    parent_id=exp_node.node_id,
                )

                # Group by group_label (epoch group)
                group_col = 'group_label' if 'group_label' in df_cell.columns else None
                if group_col:
                    for group_label, df_group in df_cell.groupby(group_col, sort=True):
                        group_protocol = ''
                        if 'protocol_name' in df_group.columns:
                            protos = df_group['protocol_name'].unique()
                            group_protocol = protos[0] if len(protos) > 0 else ''

                        group_node = _TreeNode(
                            node_id=f"grp:{exp_name}:{cell_label}:{group_label}",
                            label=f"{group_label}" + (f" ({self._short_protocol(group_protocol)})" if group_protocol else ""),
                            level='epoch_group',
                            data={'exp_name': exp_name, 'cell_label': cell_label,
                                  'group_label': group_label, 'protocol_name': group_protocol},
                            parent_id=cell_node.node_id,
                        )

                        # Each row in df_group is an epoch block
                        for _, eb_row in df_group.iterrows():
                            block_id = int(eb_row.get('block_id', 0))
                            protocol = eb_row.get('protocol_name', '')
                            block_node = _TreeNode(
                                node_id=f"blk:{exp_name}:{block_id}",
                                label=f"Block {block_id} ({self._short_protocol(protocol)})",
                                level='epoch_block',
                                data={'exp_name': exp_name, 'block_id': block_id,
                                      'protocol_name': protocol},
                                parent_id=group_node.node_id,
                            )

                            # Add epoch leaves from cache
                            epoch_key = (exp_name, block_id)
                            if epoch_key in self._epoch_cache:
                                df_epochs = self._epoch_cache[epoch_key]
                                for e_idx in range(len(df_epochs)):
                                    # Build a short param summary
                                    param_parts = []
                                    for col in df_epochs.columns:
                                        if col in ('preTime', 'stimTime', 'tailTime',
                                                    'exp_name', 'group_label',
                                                    'protocol_name', 'frame_times_ms',
                                                    'epoch_parameters', 'experiment_id',
                                                    'group_id', 'block_id', 'protocol_id',
                                                    'epoch_id'):
                                            continue
                                        val = df_epochs.iloc[e_idx].get(col)
                                        if val is not None:
                                            param_parts.append(f"{col}={val}")
                                    param_str = ", ".join(param_parts[:3])

                                    epoch_node = _TreeNode(
                                        node_id=f"epoch:{exp_name}:{block_id}:{e_idx}",
                                        label=f"Epoch {e_idx}" + (f" ({param_str})" if param_str else ""),
                                        level='epoch',
                                        data={'exp_name': exp_name, 'block_id': block_id,
                                              'epoch_idx': e_idx},
                                        parent_id=block_node.node_id,
                                    )
                                    block_node.children.append(epoch_node)
                            else:
                                # Placeholder — epochs load on expand
                                placeholder = _TreeNode(
                                    node_id=f"loading:{exp_name}:{block_id}",
                                    label="(click block to load epochs...)",
                                    level='epoch',
                                    data={},
                                    parent_id=block_node.node_id,
                                )
                                block_node.children.append(placeholder)

                            group_node.children.append(block_node)
                        cell_node.children.append(group_node)
                else:
                    # No group_label — put blocks directly under cell
                    for _, eb_row in df_cell.iterrows():
                        block_id = int(eb_row.get('block_id', 0))
                        protocol = eb_row.get('protocol_name', '')
                        block_node = _TreeNode(
                            node_id=f"blk:{exp_name}:{block_id}",
                            label=f"Block {block_id} ({self._short_protocol(protocol)})",
                            level='epoch_block',
                            data={'exp_name': exp_name, 'block_id': block_id,
                                  'protocol_name': protocol},
                            parent_id=cell_node.node_id,
                        )
                        cell_node.children.append(block_node)
                exp_node.children.append(cell_node)
            roots.append(exp_node)
        return roots

    def _apply_filters(self, roots):
        """Remove nodes that don't match the current filters. Returns pruned copy."""
        protocol_filter = self.state.protocol_filter.strip()
        match_mode = self.state.protocol_match_mode
        celltype_filter = set(self.state.celltype_filter) if self.state.celltype_filter else None
        rec_tech_filter = set(self.state.recording_technique_filter) if self.state.recording_technique_filter else None

        def matches_protocol(node):
            if not protocol_filter:
                return True
            proto = node.data.get('protocol_name', '')
            if match_mode == 'equals':
                return self._short_protocol(proto).lower() == protocol_filter.lower()
            return protocol_filter.lower() in proto.lower()

        def filter_node(node):
            if node.level == 'epoch_block':
                if not matches_protocol(node):
                    return None
            if node.level == 'epoch_group':
                if not matches_protocol(node):
                    return None
            # Filter children recursively
            filtered_children = []
            for child in node.children:
                fc = filter_node(child)
                if fc is not None:
                    filtered_children.append(fc)
            # If a non-leaf node has no visible children after filtering, hide it too
            if node.level in ('experiment', 'cell', 'epoch_group', 'epoch_block'):
                if node.children and not filtered_children:
                    return None
            node.children = filtered_children
            return node

        return [n for n in (filter_node(r) for r in roots) if n is not None]

    @staticmethod
    def _short_protocol(name):
        """Strip the Java package prefix from a protocol name."""
        if '.' in name:
            return name.rsplit('.', 1)[-1]
        return name

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def _render_html(self, roots):
        """Render the tree as nested HTML <ul> with checkboxes."""
        lines = [
            '<style>',
            '.sc-tree ul { list-style: none; padding-left: 18px; margin: 2px 0; }',
            '.sc-tree li { margin: 1px 0; }',
            '.sc-tree label { cursor: pointer; user-select: none; font-size: 13px; }',
            '.sc-tree .node-exp { font-weight: bold; }',
            '.sc-tree .node-cell { color: #1a5276; }',
            '.sc-tree .node-epoch_group { color: #1e8449; }',
            '.sc-tree .node-epoch_block { color: #7d3c98; }',
            '.sc-tree .node-epoch { color: #333; }',
            '.sc-tree details { margin: 1px 0; }',
            '.sc-tree summary { cursor: pointer; font-size: 13px; }',
            '.sc-tree input[type=checkbox] { margin-right: 4px; }',
            '.sc-tree .load-btn { font-size: 11px; color: #2980b9; cursor: pointer; ',
            '  background: none; border: 1px solid #2980b9; border-radius: 3px; padding: 1px 6px; margin-left: 6px; }',
            '</style>',
            '<div class="sc-tree">',
        ]

        def render_node(node, depth=0):
            css_class = f"node-{node.level}" if node.level != 'experiment' else 'node-exp'
            node_id_safe = node.node_id.replace('"', '&quot;')
            is_leaf = node.level == 'epoch'
            is_placeholder = node.node_id.startswith('loading:')

            if is_placeholder:
                # Show a "load" button for epoch blocks
                parent_data = node.parent_id or ''
                lines.append(
                    f'<li><button class="load-btn" '
                    f'onclick="loadEpochs(\'{parent_data}\')">'
                    f'{node.label}</button></li>'
                )
                return

            if is_leaf:
                lines.append(
                    f'<li><label class="{css_class}">'
                    f'<input type="checkbox" class="epoch-cb" '
                    f'data-node-id="{node_id_safe}" '
                    f'onchange="updateSelection(this)"/> '
                    f'{node.label}</label></li>'
                )
            else:
                has_children = bool(node.children)
                open_attr = ' open' if depth < 1 else ''
                lines.append(f'<details{open_attr}>')
                lines.append(
                    f'<summary class="{css_class}">'
                    f'<input type="checkbox" class="group-cb" '
                    f'data-node-id="{node_id_safe}" '
                    f'onchange="toggleChildren(this)"/> '
                    f'{node.label}</summary>'
                )
                if has_children:
                    lines.append('<ul>')
                    for child in node.children:
                        render_node(child, depth + 1)
                    lines.append('</ul>')
                lines.append('</details>')

        lines.append('<ul>')
        for root in roots:
            render_node(root)
        lines.append('</ul>')

        # JS for checkbox propagation and selection bridging
        bridge_class = self._bridge_id
        lines.append(f'''
<script>
function _scTreeGetBridge() {{
    var el = document.querySelector('.{bridge_class} input[type=text]');
    if (!el) el = document.querySelector('.{bridge_class}');
    return el;
}}

function toggleChildren(cb) {{
    var parent = cb.closest('details');
    if (!parent) return;
    var children = parent.querySelectorAll('input[type=checkbox]');
    for (var i = 0; i < children.length; i++) {{
        children[i].checked = cb.checked;
    }}
    updateSelection();
}}

function updateSelection(cb) {{
    var all = document.querySelectorAll('.sc-tree .epoch-cb:checked');
    var ids = [];
    for (var i = 0; i < all.length; i++) {{
        ids.push(all[i].getAttribute('data-node-id'));
    }}
    var bridge = _scTreeGetBridge();
    if (bridge) {{
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        nativeInputValueSetter.call(bridge, JSON.stringify(ids));
        bridge.dispatchEvent(new Event('input', {{ bubbles: true }}));
        bridge.dispatchEvent(new Event('change', {{ bubbles: true }}));
    }}
}}

function loadEpochs(blockNodeId) {{
    var bridge = _scTreeGetBridge();
    if (bridge) {{
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        nativeInputValueSetter.call(bridge, JSON.stringify({{"load": blockNodeId}}));
        bridge.dispatchEvent(new Event('input', {{ bubbles: true }}));
        bridge.dispatchEvent(new Event('change', {{ bubbles: true }}));
    }}
}}
</script>
''')

        lines.append('</div>')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _rebuild(self, event=None):
        """Rebuild the full tree HTML."""
        roots = self._build_tree_data()
        roots = self._apply_filters(roots)
        html = self._render_html(roots)
        self._html_pane.object = html

    def _on_selection_bridge(self, event):
        """Handle selection changes and load requests from JS."""
        import json
        try:
            data = json.loads(event.new)
        except (json.JSONDecodeError, TypeError):
            return

        if isinstance(data, dict) and 'load' in data:
            self._handle_load_request(data['load'])
            return

        # data is a list of epoch node IDs like "epoch:exp_name:block_id:epoch_idx"
        epochs = []
        for node_id in data:
            if not node_id.startswith('epoch:'):
                continue
            parts = node_id.split(':')
            if len(parts) >= 4:
                exp_name = parts[1]
                block_id = int(parts[2])
                epoch_idx = int(parts[3])
                epochs.append((exp_name, block_id, epoch_idx))
        self.state.selected_epochs = epochs

    def _handle_load_request(self, block_node_id):
        """Load epochs for a block when user clicks the load button."""
        # block_node_id is like "blk:exp_name:block_id"
        if not block_node_id.startswith('blk:'):
            return
        parts = block_node_id.split(':')
        if len(parts) < 3:
            return
        exp_name = parts[1]
        block_id = int(parts[2])

        if (exp_name, block_id) not in self._epoch_cache:
            try:
                sb, rb = self.state.get_or_load_block(exp_name, block_id, b_spiking=False)
                self._epoch_cache[(exp_name, block_id)] = sb.df_epochs
            except Exception:
                return
        self._rebuild()

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("### Data Browser", margin=(0, 5)),
            self._html_pane,
            self._selection_bridge,
            sizing_mode='stretch_width',
            scroll=True,
            max_height=500,
        )
