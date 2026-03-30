import React, { Component, useEffect } from 'react';
import { RichTreeView } from '@mui/x-tree-view/RichTreeView';
import {Alert, Snackbar} from '@mui/material/Snackbar';
import { ButtonGroup, Button,
    FormGroup, InputGroup, ControlGroup } from '@blueprintjs/core';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import CustomTreeItem from './CustomTreeItem';
import { useTreeViewApiRef } from '@mui/x-tree-view/hooks';

import axios from 'axios';

export default function ResultsTree(props){
    const [items, setItems] = React.useState(null);
    const [focusedItem, setFocusedItem] = React.useState(null);
    const [selectedItems, setSelectedItems] = React.useState([]);
    const [tag, setTag] = React.useState("");
    const [open, setOpen] = React.useState(false);
    const [error, setError] = React.useState(null);
    const [response, setResponse] = React.useState(null);

    const handleSelectedItemsChange = (event, ids) => {
        setSelectedItems(ids);
    };

    const handleTagChange = (value) => {
        setTag(value);
    }

    useEffect(() => {
        setItems(updateItems(props.results));
    }, [props.results]);

    const getAllItems = () => {
        const ids = [];
        const traverse = (item) => {
            ids.push(item.id);
            item.children.forEach((child) => traverse(child));
        }
        items.forEach((item) => traverse(item));
        return ids;
    }

    const getImmediateChildren = (id) => {
        const ids = [];
        const push = (item) => {
            ids.push(item.id);
        }
        const traverse = (item) => {
            if (item.id === id) {
                item.children.forEach((child) => push(child));
            } else {
                item.children.forEach((child) => traverse(child));
            }
        }
        items.forEach((item) => traverse(item));
        return ids;
    }

    const updateItems = (results) => {
        let items = [];
        results.forEach((result) => {
            let object = {};
            if (result.level === "experiment") {
                object['id'] = result.id + "-" + result.level + "-"  + result.id;
                object['is_mea'] = result.is_mea;
            } else {
                object['id'] = result.experiment_id + "-" + result.level + "-"  + result.id;
            }
            if (result.level === "epoch_group" || result.level === "epoch_block") {
                object['protocol'] = result.protocol;
            }
            object['level'] = result.level;
            object['dj_id'] = result.id;
            object['tags'] = result.tags;
            object['label'] = result.label ? result.label : result.id.toString();
            object['children'] = updateItems(result.children);
            items.push(object);
        });
        console.log(items);
        return items
    }

    const apiRef = useTreeViewApiRef();

    const handleFocus = (event, item) => {
        setFocusedItem(item);
        props.onFocus(apiRef.current.getItem(item).level, apiRef.current.getItem(item).dj_id);
    }

    const handleAddTags = () => {
        axios.post('http://localhost:3000/api/results/add-tags', {
            ids: selectedItems,
            tag: tag
        })
            .then(response => {console.log(response.data);})
            .catch(error => {console.log(error.response.data.message);});
    }

    const handleDeleteTags = () => {
        axios.post('http://localhost:3000/api/results/delete-tags', {
            ids: selectedItems,
            tag: tag
        })
            .then(response => {console.log(response.data);})
            .catch(error => {console.log(error.response.data.message);});
    }

    const handleSelectClick = () => {
        setSelectedItems((oldSelected) =>
          oldSelected.length === 0 ? getAllItems() : [],
        );
      };
    
    const handleSelectChildrenClick = () => {
        setSelectedItems((oldSelected) => {
            if (focusedItem == null) {
                return oldSelected;
            }
            return [...new Set([...oldSelected, ...getImmediateChildren(focusedItem)])];
        }
        );
    }

    const handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setOpen(false);
    }

    return (
        <div>
        {/* <Snackbar open={open} autoHideDuration={6000} onClose={handleClose}>
            <Alert
            onClose={handleClose}
            severity={error != null ? "error" : "success"}
            variant="filled"
            sx={{ width: '100%' }}
            >
            {error != null ? error : response}
            </Alert>
        </Snackbar> */}
            <div>
            <ControlGroup>
                <InputGroup small={true} type="text" value={tag} onValueChange={handleTagChange} placeholder="Tag to delete/add" />
                <ButtonGroup outlined={true}>
                    <Button small={true} icon="plus" intent='success'
                    disabled={selectedItems.length === 0 || tag == ""} onClick={handleAddTags}>
                    {selectedItems.length} tags</Button>
                    <Button small={true} icon="trash" intent='danger'
                    disabled={selectedItems.length === 0 || tag == ""} onClick={handleDeleteTags}>
                    {selectedItems.length} tags</Button>
                </ButtonGroup>
            </ControlGroup>
            <ButtonGroup outlined={true}>
                <Button small={true} intent={selectedItems.length === 0 ? 'primary' : 'danger'}
                onClick={handleSelectClick}>
                {selectedItems.length === 0 ? 'Select all' : 'Unselect all'}
                </Button>
                <Button small={true} intent='primary'
                disabled={focusedItem == null} onClick={handleSelectChildrenClick}>
                Select children of {focusedItem || 'focused item'}
                </Button>
            </ButtonGroup>
            </div>
            {items != null && <RichTreeView multiSelect checkboxSelection 
            apiRef={apiRef}
            selectedItems={selectedItems}
            onSelectedItemsChange={handleSelectedItemsChange}
            slots={{expandIcon: AddIcon, collapseIcon: RemoveIcon, item: CustomTreeItem}}
            items={items} expansionTrigger="iconContainer"
            onItemClick={handleFocus}/>}
        </div>
    );
}
