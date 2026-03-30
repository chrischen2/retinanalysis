import React, { Component } from 'react';
import axios from 'axios';
// import dynamic from "next/dynamic";

import AddIcon from '@mui/icons-material/Add';
import { Button, ButtonGroup, Checkbox, ControlGroup, InputGroup, Popover, Menu, MenuItem } from '@blueprintjs/core';
import { Alert, Snackbar, CircularProgress,
    Accordion, AccordionSummary, AccordionDetails,
    FormGroup} from '@mui/material';
import { Modal, Box } from '@mui/material';
import ReactJson from '@microlink/react-json-view';

import LogicBlock from './LogicBlock';

const style = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 600,
    overflow: 'scroll',
    maxHeight: '80%',
    bgcolor: 'background.paper',
    border: '2px solid #000',
    boxShadow: 24,
    p: 4,
  };

//   const DynamicReactJson = dynamic(import('@microlink/react-json-view').then((mod) => mod.default), { 
//     ssr: false,
//     loading: () => <p>Loading...</p>,
// });

class QueryContainer extends Component {
    constructor(props) {
    super(props);
    this.state = {
        error: null,
        response: null,
        levels: null,
        fields: null,
        tag_fields: null,
        open: false,
        query_obj: {},
        final_query: {},
        injected_query: {},
        set_query: "",
        queries: {},
        exclude_levels: [],
        query_name: "",
        hideExclude: false,
        modalOpen: false
    };
    }

    exclude_obj = {
        "NOT": [{
            "COND": {
                type : "TAG",
                value : "tag='exclude'"
                }
        }]
    }

    getLevelsAndFields = () => {
    axios.get('http://localhost:3000/api/query/get-levels-and-fields')
        .then(response => {
        // Handle success by setting the data in the state
        this.setState({ levels: response.data.levels });
        this.setState({ fields: response.data.fields });
        this.setState({ tag_fields: response.data.tag_fields });
        })
        .catch(error => {
        // Handle error by setting the error message in the state
            this.setState({ error: error.response.data.message, response: null, open: true});
        });
    }

    componentDidMount() {
        this.getLevelsAndFields();
        this.getQueryList();
    }
    
    handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        this.setState({ open: false });
    }

    handleAfterEffects = (temp_obj, excluded) => {
        if (excluded) {
            if ("epoch" in temp_obj) {
                let inner_obj = temp_obj["epoch"];
                temp_obj["epoch"] = { "AND":
                    [inner_obj, this.exclude_obj]
                 }
            } else {
                temp_obj["epoch"] = this.exclude_obj;
            }
        }
        this.setState({ final_query: temp_obj });
        // this.setState({ injected_query: {} });
        this.props.onQueryObj(temp_obj);
    }

    handleQueryChange = (query, table_name) => {
        let temp_obj = {};
        if (Object.keys(query).length == 0) {
            temp_obj = this.state.query_obj;
            delete temp_obj[table_name];
        } else {
            temp_obj = { ...this.state.query_obj, [table_name]: query };
        }
        this.setState({ query_obj: temp_obj });
        this.handleAfterEffects(structuredClone(temp_obj), this.state.hideExclude);
    }

    handleModalOpen = () => {
        this.setState({ modalOpen: true });
    }

    handleModalClose = () => {
        this.setState({ modalOpen: false });
    }

    handleLevelSelect = (event) => {
        if (event.target.checked) {
            if (this.state.exclude_levels.includes(event.target.value)) {
                let temp_list = this.state.exclude_levels.filter(e => e !== event.target.value);
                this.props.onExcludeChange(temp_list);
                this.setState({ exclude_levels: temp_list });
            }
        } else {
            if (!(this.state.exclude_levels.includes(event.target.value))) {
                let temp_list = this.state.exclude_levels;
                temp_list.push(event.target.value);
                this.props.onExcludeChange(temp_list);
                this.setState({ exclude_levels: temp_list });
            }
        }
    }

    handleNameChange = (value) => {
        this.setState({ query_name: value });
    }

    handleQueryInject = (set_query, query) => {
        this.handleAfterEffects(structuredClone(query), this.state.hideExclude);
        this.setState({ injected_query: {}, set_query: "wipe" }, () => {
            this.setState({ injected_query: query, set_query: set_query });
        });
        this.setState({ query_obj: query });
    }

    handleQueryClear = () => {
        this.handleAfterEffects({}, this.state.hideExclude);
        this.setState({ injected_query: {}, set_query: "wipe" }, () => {
            this.setState({ injected_query: {}, set_query: "empty" })
        });
        this.setState({ query_obj: {} });
    }

    getQueryList = () => {
        axios.get('http://localhost:3000/api/query/get-saved-queries')
        .then(response => {
            this.setState({ queries: response.data.queries });
        })
        .catch(error => {
            this.setState({ error: error.response.data.message, response: null, open: true});
        });
    }

    handleAddQuery = () => {
        axios.post('http://localhost:3000/api/query/add-saved-query', {
            query_name: this.state.query_name,
            query_obj: this.state.query_obj
        })
        .then(response => {
            this.setState({ response: response.data.message, error: null, open: true });
            this.getQueryList();
        })
        .catch(error => {
            this.setState({ error: error.response.data.message, response: null, open: true });
        });
    }

    handleDeleteQuery = (query_name) => {
        axios.post('http://localhost:3000/api/query/delete-saved-query', {
            query_name: query_name
        })
        .then(response => {
            this.setState({ response: response.data.message, error: null, open: true });
            this.getQueryList();
        })
        .catch(error => {
            this.setState({ error: error.response.data.message, response: null, open: true });
        });
    }

    handleExcludeClick = () => {
        this.handleAfterEffects(structuredClone(this.state.query_obj), !this.state.hideExclude);
        this.setState({ hideExclude: !this.state.hideExclude});
    }

    render() {
    const { error, response, open, injected_query, query_name, set_query, queries, query_obj,
        levels, fields, tag_fields, modalOpen, hideExclude } = this.state;

    return (
        <div>
        <Snackbar open={open} autoHideDuration={6000} onClose={this.handleClose}>
            <Alert
            onClose={this.handleClose}
            severity={error != null ? "error" : "success"}
            variant="filled"
            sx={{ width: '100%' }}
            >
            {error != null ? error : response}
            </Alert>
        </Snackbar>

        <Modal
        open={modalOpen}
        onClose={this.handleModalClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
        >
        <Box sx={style}>
        <ReactJson src={this.state.final_query} />
        </Box>
        </Modal>

        {levels != null ? <div style={{ margin: "0%", padding: "0%" }}>
        {levels.map((level, index) => (
            <Accordion key={index} style={{ margin: "0%" }} 
            sx={{'& .MuiAccordionSummary-content': {margin: "2px 0"},
            '& .MuiAccordionSummary-content.Mui-expanded': {margin: "4px 0"},
            '& .MuiAccordionDetails-root': {padding: "1%"},
        }}>
            <AccordionSummary expandIcon={<AddIcon />} style={{ minHeight: "0" }}>
                <Checkbox onChange={this.handleLevelSelect} label={level} value={level} defaultChecked={true} />
            </AccordionSummary>
            <AccordionDetails>
                <LogicBlock key={level}
                table_name={level}
                fields={fields[level]}
                tag_fields={tag_fields}
                query={injected_query[level] ? injected_query[level] : null}
                set_query={set_query}
                onQueryChange={this.handleQueryChange} />
            </AccordionDetails>
            </Accordion>
        ))}
        <FormGroup>
        <Checkbox
            checked={hideExclude}
            onChange={this.handleExcludeClick}
            label='hide excluded'/>
        <ControlGroup>
            <InputGroup small={true} type="text" value={query_name} onValueChange={this.handleNameChange} placeholder="Name for query" />
            <ButtonGroup outlined={true}>
                <Button small={true} icon="add" intent='success'
                disabled={query_name == ""} onClick={this.handleAddQuery}>
                Save current query</Button>
            </ButtonGroup>
        </ControlGroup>
        <ButtonGroup>
            <Button small={true} onClick={this.handleModalOpen}>Peek at QueryObj</Button>
            <Button small={true} icon="eraser" intent="warning" disabled={Object.keys(query_obj).length == 0}
            onClick={this.handleQueryClear}>Clear</Button>
            <Popover content={<Menu>
                {Object.keys(queries).map((key, index) => (
                    <MenuItem key={key} 
                    icon="application"
                    text={key}
                    children={<>
                        <MenuItem text="inject ..." intent='primary' icon="bring-data"
                        onClick={() => this.handleQueryInject(key, this.state.queries[key])}/>
                        <MenuItem text="delete ..." intent='danger' icon="trash"
                        onClick={() => this.handleDeleteQuery(key)}/>
                    </>}/>
                ))}
            </Menu>} placement="bottom-start">
                <Button alignText="left" disabled= {Object.keys(queries).length == 0} small={true}
                icon="applications" rightIcon="caret-down" text="Saved queries ..." />
            </Popover>
        </ButtonGroup>
        </FormGroup>
        {/* {isLoading ? <CircularProgress /> : null} */}
    </div> : <p>Loading...</p>}
        </div>
    );
    }
}

export default QueryContainer;
