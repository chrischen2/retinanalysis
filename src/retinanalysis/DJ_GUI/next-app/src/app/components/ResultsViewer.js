"use client"

import { useState } from 'react';
// import dynamic from "next/dynamic";
import axios from 'axios';
import ReactJson from '@microlink/react-json-view';
import ResultsTree from './results/ResultsTree';
import Information from './results/Information';
import { Button, AnchorButton, ButtonGroup, Popover, 
    Code, Callout, MenuDivider, Menu, MenuItem } from '@blueprintjs/core';
import { Snackbar, Alert } from '@mui/material';


export default function ResultsViewer(props){
  const [focusedData, setFocusedData] = useState(null);
  const [level, setLevel] = useState(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
//   const DynamicReactJson = dynamic(import('@microlink/react-json-view').then((mod) => mod.default), { 
//         ssr: false,
//         loading: () => <p>Loading...</p>,
//     });

    const results = props.results != null ? props.results : null

    const getFocusedData = (level, id) => {
        axios.post('http://localhost:3000/api/results/get-metadata',
            { level: level, id: id }
        )
        .then(response => {
            setFocusedData(response.data.metadata);
        })
        .catch(error => {
            console.log(error.response.data.message);
        });
    }

    const displayInfo = (level, id) => {
        getFocusedData(level, id);
        setLevel(level);
    }

    const handlePushTags = () => {
        axios.post('http://localhost:3000/api/results/push-tags',
            { experiment_ids: results.map(result => result.object[0].id) }
        )
        .then(response => {
            console.log(response.data);
        })
        .catch(error => {
            console.log(error.response.data.message);
        });
    }

    const handlePullTags = () => {
        axios.post('http://localhost:3000/api/results/pull-tags',
            { experiment_ids: results.map(result => result.object[0].id) }
        )
        .then(response => {
            console.log(response.data);
        })
        .catch(error => {
            console.log(error.response.data.message);
        });
    }

    const handleResetTags = () => {
        axios.post('http://localhost:3000/api/results/reset-tags',
            { experiment_ids: results.map(result => result.object[0].id) }
        )
        .then(response => {
            console.log(response.data.message);
        })
        .catch(error => {
            console.log(error.response.data.message);
        });
    }

    const handleDownload = (include_levels, include_meta) => {
        axios.post('http://localhost:3000/api/results/download-results',
            {exclude_levels: !include_levels,
            include_meta: include_meta }
        )
        .then(response => {
            console.log(response.data);
            setOpen(true);
            setResponse(response.data.message);
            setError(null);
        })
        .catch(error => {
            console.log(error.response.data.message);
            setOpen(true);
            setError(error.response.data.message);
            setResponse(null);
        });
    }

    const handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setOpen(false);
    }

    return (
    <div style={{ display: 'flex', height: '100%'}}>
        <Snackbar open={open} autoHideDuration={6000} onClose={handleClose}>
            <Alert
            onClose={handleClose}
            severity={error != null ? "error" : "success"}
            variant="filled"
            sx={{ width: '100%' }}
            >
            {error != null ? error : response}
            </Alert>
        </Snackbar>
        <div style={{ flex: 1}}>
            <div style={{ marginRight: "1%", 
            backgroundColor: "darkgray", borderRadius: "1%", padding: "1%",
            overflowX: "auto", overflowY: "auto", height: '95%' }}>
                {results && <ResultsTree results={results} onFocus={displayInfo} />}
            </div>
            <ButtonGroup style={{ marginRight: "1%", padding: "1%", height: '5%' }}>
            {results && <Popover content={
                <Menu>
                    <MenuDivider title="Write to folder:"/>
                    <MenuItem text="Raw tree" 
                        onClick={() => handleDownload(false, false)}/>
                    <MenuItem text="Include metadata" 
                        onClick={() => handleDownload(false, true)}/>
                    <MenuItem text="Include all levels" 
                        onClick={() => handleDownload(true, false)}/>
                    <MenuItem text="Include all levels + metadata" 
                        onClick={() => handleDownload(true, true)}/>
                    <MenuDivider title="In-browser:"/>
                    <MenuItem text="Raw tree"
                        href={`data:text/json;charset=utf-8,${encodeURIComponent(
                            JSON.stringify(results, null, '\t')
                          )}`}
                          download="results.json"/>
                </Menu>
            } placement="bottom-start">
                <Button rightIcon="caret-down" icon="cloud-download" text="Download ..." />
            </Popover>
            }
            
            { results && results[0].level == "experiment" &&
            <>
            <Button disabled={true}>Tag Options:</Button>
            <Popover enforceFocus={false} placement='top' interactionKind='hover'
                content={
                    <Callout intent="warning" compact={true}>
                        Export tags with your username to <Code>/tags/</Code>
                    </Callout>
                }>
            <Button onClick={handlePushTags}>Push</Button>
            </Popover>
            <Popover enforceFocus={false} placement='top' interactionKind='hover'
                content={
                    <Callout intent="primary" compact={true}>
                        Import tags from <Code>/tags/</Code> made by other users
                    </Callout>
                }>
                <Button onClick={handlePullTags}>Pull</Button>
            </Popover>
            <Popover enforceFocus={false} placement='top' interactionKind='hover'
                content={
                    <Callout intent="danger" compact={true}>
                        Import tags from <Code>/tags/</Code> made by all users
                    </Callout>
                }>
                <Button onClick={handleResetTags}>Reset</Button>
            </Popover>
            </>
            }
            </ButtonGroup>
        </div>
        <div style={{ flex: 1, marginLeft: "1%", maxWidth: "48%"}}>
            <div style={{ height: "55%", overflow: "auto",
                borderRadius: "1%", backgroundColor: "darkgrey",
                 margin: "0% 1% 3% 0%" }}>
                {!focusedData ? <div>Click on a valid item in the tree to view visualizations</div> :
                <Information level={level} id={focusedData.id} experiment_id={focusedData.experiment_id} />}
            </div>
            <div style={{ height: "43%", overflow: "auto",
                borderRadius: "1%", backgroundColor: "#1e1e1e", 
                margin: "3% 1% 1% 0%"}}>
                {focusedData != null &&
                <ReactJson style={{margin: "3%"}} theme={'twilight'} src={focusedData} collapsed="true" />}
            </div>
        </div>
    </div>
    );
}
