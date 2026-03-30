import * as React from 'react';

import { styled } from '@mui/material/styles';
import Chip from '@mui/material/Chip';
import { CompoundTag } from '@blueprintjs/core';
import Stack from '@mui/material/Stack';
import { unstable_useTreeItem2 as useTreeItem2 } from '@mui/x-tree-view/useTreeItem2';
import {
  TreeItem2Content,
  TreeItem2IconContainer,
  TreeItem2GroupTransition,
  TreeItem2Label,
  TreeItem2Root,
  TreeItem2Checkbox,
} from '@mui/x-tree-view/TreeItem2';
import { TreeItem2Icon } from '@mui/x-tree-view/TreeItem2Icon';
import { TreeItem2Provider } from '@mui/x-tree-view/TreeItem2Provider';
import { TreeItem2DragAndDropOverlay } from '@mui/x-tree-view/TreeItem2DragAndDropOverlay';
import Box from '@mui/material/Box';

const CustomTreeItemContent = styled(TreeItem2Content)(({ theme }) => ({
    padding: theme.spacing(0.5, 1),
  }));

const CustomTreeItem = React.forwardRef(function CustomTreeItem(props, ref) {
    const { id, itemId, label, disabled, children, ...other } = props;
  
    const {
      getRootProps,
      getContentProps,
      getIconContainerProps,
      getCheckboxProps,
      getLabelProps,
      getGroupTransitionProps,
      getDragAndDropOverlayProps,
      status,
      publicAPI
    } = useTreeItem2({ id, itemId, children, label, disabled, rootRef: ref });

    const item = publicAPI.getItem(itemId);
  
    return (
      <TreeItem2Provider itemId={itemId}>
        <TreeItem2Root {...getRootProps(other)}>
          <CustomTreeItemContent {...getContentProps()}>
            <TreeItem2IconContainer {...getIconContainerProps()}>
              <TreeItem2Icon status={status} />
            </TreeItem2IconContainer>
            <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
              <TreeItem2Checkbox {...getCheckboxProps()} />
              <TreeItem2Label {...getLabelProps()} >
              <Stack direction="row" spacing={1}>
              <Chip label={item.level} size="small"/>
              {item.level == 'experiment' &&
                  <Chip label={item.is_mea == 1 ? 'mea' : 'patch'} size="small" color="primary"/>}
              {(item.level == 'epoch_group' || item.level == 'epoch_block') 
                && item.protocol && item.protocol != 'no_group_protocol'
                && <Chip label={item.protocol.split('.').pop()} size="small" color="secondary"/>}
              </Stack>
              <Stack direction="row" spacing={1}>
                <div>{label}</div>
              </Stack>
              <Stack direction="row" spacing={1}>
                {
                  item.tags.map((tag) => {
                    return <CompoundTag key={tag.tag_id} leftContent={tag.tag}
                                        round={true}>
                      {tag.user}
                    </CompoundTag>
                  })
                }
              </Stack>
              </TreeItem2Label>
            </Box>
            <TreeItem2DragAndDropOverlay {...getDragAndDropOverlayProps()} />
          </CustomTreeItemContent>
          {children && <TreeItem2GroupTransition {...getGroupTransitionProps()} />}
        </TreeItem2Root>
      </TreeItem2Provider>
    );
  });

export default CustomTreeItem;