import React, { Component } from 'react';
import axios from 'axios';

import { Alert, Snackbar, CircularProgress} from '@mui/material';
import { ButtonGroup, Button } from '@blueprintjs/core';
import CondBlock from './CondBlock';

class LogicBlock extends Component {
    constructor(props) {
    super(props);
    this.state = {
        error: null,
        response: null,
        open: false,
        table_name: null,
        logic_op: "all",
        criteria_list: []
    };
    }

    // getFields = () => {
    // axios.post('http://localhost:3000/api/query/get-table-fields', { table_name: this.props.table_name })
    //     .then(response => {
    //     // Handle success by setting the data in the state
    //     console.log(response.data.fields);
    //     console.log(this.props.table_name);
    //     this.setState({ fields: response.data.fields });
    //     })
    //     .catch(error => {
    //     // Handle error by setting the error message in the state
    //         this.setState({ error: error.response.data.message, response: null, open: true});
    //     });
    // }

    componentDidUpdate(prevProps) {
        if (prevProps.set_query !=  this.props.set_query 
            // || prevProps.query != this.props.query
        ) {
            // check that current query is not empty
            if (this.props.query != null) {
                // console.log("inserting state for table", this.props.table_name);
                // console.log("state in logic block: ", this.props.query);
                if (Object.keys(this.props.query)[0] === "AND") {
                    this.setState({ logic_op: "all" });
                } else if (Object.keys(this.props.query)[0] === "OR") {
                    this.setState({ logic_op: "any" });
                } else {
                    this.setState({ logic_op: "none" });
                }
                if (this.props.query[Object.keys(this.props.query)[0]] != null) {
                    this.setState({ criteria_list: this.props.query[Object.keys(this.props.query)[0]] });
                } else {
                    this.setState({ criteria_list: [] });
                }
                this.setState({ criteria_list: this.props.query[Object.keys(this.props.query)[0]] 
                    ? this.props.query[Object.keys(this.props.query)[0]] : []
                 });
            } else { // if it is empty, wipe
                this.setState({ criteria_list: [] });
            }
        }
    }

    componentDidMount() {
        this.componentDidUpdate({query: {}, set_query: ""});
    }
    
    handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        this.setState({ open: false });
    }

    handleAddLogic = () => {
        this.setState({ criteria_list: [...(this.state.criteria_list ?? []), { "AND": [] }] });
    }

    handleAddCond = () => {
        this.setState({ criteria_list: [...(this.state.criteria_list ?? []), { "COND": {} }] });
    }

    handleLogicChange = (event) => {
        this.setState({ logic_op: event.target.value });
        // make event.target.value the key of the dictionary
        this.sendUp(event.target.value, this.state.criteria_list);
    }

    sendUp = (logic_op, criteria_list) => {
        // if criteria_list is empty, send up an empty dictionary
        if (criteria_list.length === 0) {
            this.props.onQueryChange({}, this.props.table_name);
        } else if (logic_op === "all") {
            this.props.onQueryChange({ "AND": criteria_list }, this.props.table_name);
        } else if (logic_op === "any") {
            this.props.onQueryChange({ "OR": criteria_list }, this.props.table_name);
        } else {
            this.props.onQueryChange({ "NOT": criteria_list }, this.props.table_name);
        }
    }

    render() {
    const { error, response, open, logic_op, criteria_list} = this.state;
    const { fields, tag_fields } = this.props;

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

        <div style={{ border: "lightgrey solid 1px" }}>
        {/* Dropdown to choose between all/any/none of these are true */}
        {/* store this in state */}
        <select value={logic_op} onChange={this.handleLogicChange}>
            <option value="all">All</option>
            <option value="any">Any</option>
            <option value="none">None</option>
        </select>
        &nbsp; of these are true:
        <div style = {{marginLeft : "2%"}}>
        {criteria_list &&
        criteria_list.map((criteria, index) => {
            const handleQueryChange = (query, table_name) => {
                const updatedCriteriaList = [
                    ...this.state.criteria_list.slice(0, index),
                    query,
                    ...this.state.criteria_list.slice(index + 1)
                ];
                this.setState({ criteria_list: updatedCriteriaList });
                this.sendUp(logic_op, updatedCriteriaList);
            };

            const handleDelete = () => {
                const updatedCriteriaList = [
                    ...this.state.criteria_list.slice(0, index),
                    ...this.state.criteria_list.slice(index + 1)
                ];
                this.setState({ criteria_list: updatedCriteriaList });
                this.sendUp(logic_op, updatedCriteriaList);
            };

            return (
                <div key={index}>
                    {Object.keys(criteria)[0] !== "COND" ?
                        <LogicBlock
                            key={index}
                            table_name={this.props.table_name}
                            fields={fields}
                            tag_fields={tag_fields}
                            set_query={this.props.set_query}
                            query={criteria}
                            onDelete={handleDelete}
                            onQueryChange={handleQueryChange}
                        />
                        :
                        <CondBlock
                            key={index}
                            table_name={this.props.table_name}
                            fields={fields}
                            tag_fields={tag_fields}
                            set_query={this.props.set_query}
                            query={criteria["COND"]}
                            onDelete={handleDelete}
                            onQueryChange={handleQueryChange}
                        />
                    }
                </div>
            );
        })}
        </div>
        <ButtonGroup outlined={true}>
            <Button icon="plus" small={true} intent='primary'
            onClick={this.handleAddLogic}>logical block</Button>
            <Button icon="plus" small={true} intent='primary'
            onClick={this.handleAddCond} >condition</Button>
            <Button icon="trash" small={true} intent='danger'
            onClick={this.props.onDelete} >delete</Button>
        </ButtonGroup>
        {/* {isLoading ? <CircularProgress /> : null} */}
        </div>
        </div>
    );
    }
}

export default LogicBlock;
