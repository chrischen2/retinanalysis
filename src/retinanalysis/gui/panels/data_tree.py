"""Hierarchical data tree browser.

Displays: exp_name -> Cell -> EpochGroup -> EpochBlock
Clicking a block loads all its epochs and shows raw traces.
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
        self.level = level  # 'experiment', 'cell', 'epoch_group', 'epoch_block'
        self.data = data or {}
        self.children = []
        self.parent_id = parent_id


class DataTree(pn.viewable.Viewer):
    """Hierarchical tree browser built from loaded experiment summaries.

    The tree is rendered as HTML with a clickable hierarchy.
    Clicking a block auto-loads its epochs and plots all raw traces.
    """

    state = param.ClassSelector(class_=AppState)

    def __init__(self, state, **params):
        super().__init__(state=state, **params)

        self._html_pane = pn.pane.HTML(
            "", sizing_mode='stretch_width', min_height=200,
        )
        # Bridge widget: JS writes block selection here, Python watches it
        self._selection_bridge = pn.widgets.TextInput(
            value='', visible=False,
        )
        self._selection_bridge.param.watch(self._on_selection_bridge, 'value')

        # Track which block is currently selected for highlighting
        self._selected_block_id = None

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

            # Get experiment start date if available
            exp_date = ''
            if 'start_time' in df.columns:
                first_time = df['start_time'].min()
                if first_time is not None:
                    try:
                        exp_date = first_time.strftime('%m/%d/%y, %H:%M:%S')
                    except Exception:
                        pass

            exp_label = exp_name
            if exp_date:
                exp_label = f"{exp_date}"
            if species:
                exp_label += f" ({species})"

            exp_node = _TreeNode(
                node_id=f"exp:{exp_name}",
                label=exp_label,
                level='experiment',
                data={'exp_name': exp_name},
            )

            # Group by cell_label
            if 'cell_label' not in df.columns:
                roots.append(exp_node)
                continue

            for cell_label, df_cell in df.groupby('cell_label', sort=True):
                # Get cell type if available
                cell_type = ''
                if 'cell_type' in df_cell.columns:
                    types = df_cell['cell_type'].dropna().unique()
                    if len(types) > 0 and types[0] != 'Unknown':
                        cell_type = str(types[0])

                cell_display = str(cell_label)
                if cell_type:
                    cell_display += f" ({cell_type})"

                cell_node = _TreeNode(
                    node_id=f"cell:{exp_name}:{cell_label}",
                    label=cell_display,
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
                            start_time = eb_row.get('start_time', None)
                            duration = eb_row.get('duration_minutes', None)

                            # Build block label: timestamp + protocol
                            time_str = ''
                            if start_time is not None:
                                try:
                                    time_str = start_time.strftime('%H:%M:%S')
                                except Exception:
                                    pass

                            block_label = time_str if time_str else f"Block {block_id}"
                            block_label += f"  {self._short_protocol(protocol)}"
                            if duration is not None:
                                block_label += f" ({duration:.1f}m)"

                            block_node = _TreeNode(
                                node_id=f"blk:{exp_name}:{block_id}",
                                label=block_label,
                                level='epoch_block',
                                data={'exp_name': exp_name, 'block_id': block_id,
                                      'protocol_name': protocol},
                                parent_id=group_node.node_id,
                            )
                            group_node.children.append(block_node)
                        cell_node.children.append(group_node)
                else:
                    # No group_label — put blocks directly under cell
                    for _, eb_row in df_cell.iterrows():
                        block_id = int(eb_row.get('block_id', 0))
                        protocol = eb_row.get('protocol_name', '')
                        start_time = eb_row.get('start_time', None)

                        time_str = ''
                        if start_time is not None:
                            try:
                                time_str = start_time.strftime('%H:%M:%S')
                            except Exception:
                                pass

                        block_label = time_str if time_str else f"Block {block_id}"
                        block_label += f"  {self._short_protocol(protocol)}"

                        block_node = _TreeNode(
                            node_id=f"blk:{exp_name}:{block_id}",
                            label=block_label,
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
            if node.level in ('experiment', 'cell', 'epoch_group'):
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
        """Render the tree as nested HTML with clickable blocks."""
        selected = self._selected_block_id or ''
        lines = [
            '<style>',
            '.sc-tree { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }',
            '.sc-tree ul { list-style: none; padding-left: 16px; margin: 1px 0; }',
            '.sc-tree li { margin: 1px 0; }',
            '.sc-tree details { margin: 1px 0; }',
            '.sc-tree summary { cursor: pointer; font-size: 13px; padding: 2px 4px; border-radius: 3px; }',
            '.sc-tree summary:hover { background: #f0f0f0; }',
            '.sc-tree .node-exp { font-weight: bold; font-size: 13px; }',
            '.sc-tree .node-cell { color: #1a5276; font-weight: 600; font-size: 13px; }',
            '.sc-tree .node-grp { color: #1e8449; font-size: 13px; }',
            '.sc-tree .block-item { cursor: pointer; font-size: 12px; padding: 3px 8px; ',
            '  border-radius: 3px; margin: 1px 0; user-select: none; }',
            '.sc-tree .block-item:hover { background: #e8f4fd; }',
            '.sc-tree .block-item.selected { background: #d4edfc; font-weight: 600; }',
            '.sc-tree .block-protocol { color: #7d3c98; }',
            '.sc-tree .block-time { color: #666; }',
            '.sc-tree .block-dur { color: #999; font-size: 11px; }',
            '</style>',
            '<div class="sc-tree">',
        ]

        def render_node(node, depth=0):
            node_id_safe = node.node_id.replace('"', '&quot;')

            if node.level == 'epoch_block':
                # Blocks are clickable leaf items (not collapsible)
                is_selected = node.node_id == selected
                sel_class = ' selected' if is_selected else ''
                lines.append(
                    f'<li><div class="block-item{sel_class}" '
                    f'data-node-id="{node_id_safe}" '
                    f'onclick="selectBlock(\'{node_id_safe}\')">'
                    f'{node.label}</div></li>'
                )
            else:
                # Collapsible parent nodes
                css_class = {
                    'experiment': 'node-exp',
                    'cell': 'node-cell',
                    'epoch_group': 'node-grp',
                }.get(node.level, '')

                open_attr = ' open' if depth < 2 else ''
                lines.append(f'<details{open_attr}>')
                lines.append(
                    f'<summary class="{css_class}">'
                    f'{node.label}</summary>'
                )
                if node.children:
                    lines.append('<ul>')
                    for child in node.children:
                        render_node(child, depth + 1)
                    lines.append('</ul>')
                lines.append('</details>')

        if not roots:
            lines.append('<p style="color: #999; font-size: 13px; padding: 8px;">'
                         'No experiments loaded. Use "Add Experiment" above.</p>')
        else:
            lines.append('<ul style="padding-left: 0;">')
            for root in roots:
                render_node(root)
            lines.append('</ul>')

        # JS: clicking a block sends its ID to the Python bridge
        bridge_id = self._selection_bridge.id
        lines.append(f'''
<script>
function selectBlock(blockNodeId) {{
    // Highlight the selected block
    var allBlocks = document.querySelectorAll('.sc-tree .block-item');
    for (var i = 0; i < allBlocks.length; i++) {{
        allBlocks[i].classList.remove('selected');
    }}
    var clicked = document.querySelector('.sc-tree .block-item[data-node-id="' + blockNodeId + '"]');
    if (clicked) {{
        clicked.classList.add('selected');
    }}

    // Send block ID to Python via bridge
    var bridge = document.getElementById('{bridge_id}');
    if (bridge) {{
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        nativeInputValueSetter.call(bridge, blockNodeId);
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
        """Handle block click: load the block and select all its epochs."""
        block_node_id = event.new
        if not block_node_id or not block_node_id.startswith('blk:'):
            return

        parts = block_node_id.split(':')
        if len(parts) < 3:
            return
        exp_name = parts[1]
        block_id = int(parts[2])

        # Update highlight state and rebuild tree HTML
        self._selected_block_id = block_node_id

        # Load block and select all its epochs (triggers trace viewer)
        self.state.select_block(exp_name, block_id)

        # Rebuild to update highlight
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
