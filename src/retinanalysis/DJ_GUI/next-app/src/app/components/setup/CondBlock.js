import React, { Component } from 'react';
import axios from 'axios';

import { ButtonGroup, Button, 
    HTMLSelect, SegmentedControl, Callout, Code,
    FormGroup, InputGroup, ControlGroup } from '@blueprintjs/core';
import { Alert, Snackbar, CircularProgress} from '@mui/material';

class CondBlock extends Component {
    constructor(props) {
    super(props);
    this.state = {
        error: null,
        response: null,
        open: false,
        table_name: null,
        cond_type: "PARAM",
        field_index: 0,
        subfield_name: "",
        subfield_type: "",
        type_set: true,
        cur_operators: this.props.fields ? this.props.fields[0][1] == "string" ? this.str_ops : this.int_ops : null,
        operator: "",
        value: "",
        unsaved_changes: false,
    };
    }

    int_ops = [["=", "equal to"], ["!=", "not equal to"], ["<", "less than"], 
    [">", "greater than"], ["<=", "less than or equal to"], [">=", "greater than or equal to"]];

    str_ops = this.int_ops.concat([["like", "like"], ["not like", "not like"]]);

    handleCondTypeChange = (value) => {
        this.setState({ cond_type: value });
        this.setState({ unsaved_changes: true });
        this.setState({ field_index: 0, subfield_name: "", subfield_type: "", type_set: false });
    }

    handleFieldChange = (event) => {
        this.setState({ field_index: event.target.value });
        if (this.state.cond_type == "PARAM" && this.props.fields[event.target.value][1] == "json") {
            this.setState({ type_set: false });
        } else {
            this.setState({ subfield_type: "", subfield_name: "" });
            this.setState({ type_set: true });
            let field_arr = this.state.cond_type == "PARAM" ? this.props.fields : this.props.tag_fields;
            this.setState({ cur_operators: 
                field_arr[event.target.value][1] == "string" 
                ? this.str_ops : this.int_ops });
        }
        this.setState({ unsaved_changes: true });
    }

    handleSubfieldChange = (value) => {
        this.setState({ subfield_name: value });
        this.setState({ type_set: false, subfield_type: "", cur_operators: null });
        this.setState({ unsaved_changes: true });
    }

    handleSubfieldTypeChange = (value) => {
        this.setState({ subfield_type: value });
        this.setState({ type_set: true });
        this.setState({ cur_operators: value == "string" ? this.str_ops : this.int_ops });
        this.setState({ unsaved_changes: true });
    }

    handleOperatorChange = (event) => {
        this.setState({ operator: event.target.value });
        this.setState({ unsaved_changes: true });
    }

    handleValueChange = (value) => {
        this.setState({ value: value });
        this.setState({ unsaved_changes: true });
    }

    // method to reverse engineer the query object and populate the state!
    componentDidUpdate(prevProps) {
        if (Object.keys(this.props.query).length > 0
            && prevProps.set_query !=  this.props.set_query) {
            // console.log("setting state from query object", this.props.query);
            this.setState({ cond_type: this.props.query.type });
            // first, detect whether this is a json param field, other param field, or tag field
            // based on query.value, which is constructed as shown in sendUp below
            let field_name, field_index, subfield_name, subfield_type, cur_operators;
            // if value contains "->>'$.", it is a json field.
            if (this.props.query.value.includes("->>'$.")) {
                // extract the field name and subfield name
                field_name = this.props.query.value.split("->>'$")[0];
                // get field_index from field_name, which is the index in props.fields[0]
                field_index = this.props.fields.findIndex((field) => field[0] == field_name);
                // subfield is between "->>'$." and "'"
                subfield_name = this.props.query.value.split("->>'$.")[1].split("'")[0];
                // if last character of query.value is ', it is a string field
                if (this.props.query.value.slice(-1) == "'") {
                    subfield_type = "string";
                    cur_operators = this.str_ops;
                } else {
                    subfield_type = "number";
                    cur_operators = this.int_ops;
                }
            } else if (this.props.query.type == "PARAM") {
                field_name = this.props.query.value.split(" ")[0];
                field_index = this.props.fields.findIndex((field) => field[0] == field_name);
                cur_operators = this.props.fields[field_index][1] == "string" ? this.str_ops : this.int_ops;
            } else {
                field_name = this.props.query.value.split(" ")[0];
                field_index = this.props.tag_fields.findIndex((field) => field[0] == field_name);
                cur_operators = this.props.tag_fields[field_index][1] == "string" ? this.str_ops : this.int_ops;
            }
            let split_str = this.props.query.value.split(" ");
            let operator = split_str.slice(1, -1).join(" ");
            let value = split_str.slice(-1)[0];
            if (value[0] == "'" && value.slice(-1) == "'") {
                value = value.slice(1, -1);
            }
            // populate all states that apply.
            this.setState({ field_index: field_index, type_set: true, unsaved_changes: false });
            if (subfield_name) { // json field
                this.setState({ subfield_name: subfield_name, subfield_type: subfield_type });
            }
            this.setState({ operator: operator, value: value, cur_operators: cur_operators });
        }
    }

    componentDidMount() {
        this.componentDidUpdate({query: {}, set_query: ""});
    }

    sendUp = () => {
        if (this.state.operator && this.state.value) {
            // Need to construct value. First, adding the parameter: if it is a json field, formatted slightly different.
            let condition = "";
            if (this.state.cond_type == "PARAM" && this.props.fields[this.state.field_index][1] == "json") {
                condition = this.props.fields[this.state.field_index][0] + "->>'$."
                            + this.state.subfield_name + "'";
            } else if (this.state.cond_type == "PARAM"){
                condition = this.props.fields[this.state.field_index][0];
            } else {
                condition = this.props.tag_fields[this.state.field_index][0];
            }
            // Adding the operator and value
            condition += " " + this.state.operator + " ";
            if ((this.state.subfield_type && this.state.subfield_type == "string") 
                || (this.state.cond_type == "PARAM" && this.props.fields[this.state.field_index][1] == "string")
                || (this.state.cond_type == "TAG" && this.props.tag_fields[this.state.field_index][1] == "string")) {
                condition += "'" + this.state.value + "'";
            } else {
                condition += this.state.value;
            }
            this.setState({ unsaved_changes: false });
            this.props.onQueryChange({ "COND": { "type": this.state.cond_type, "value": condition } }, this.props.table_name);
        }
    }
    
    handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        this.setState({ open: false });
    }

    render() {
    const { error, response, open, cond_type, cur_operators, operator, value,
        field_index, subfield_name, subfield_type, type_set, unsaved_changes} = this.state;
    const { fields, tag_fields, table_name } = this.props;

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
            {fields && tag_fields ? <div> 
                <FormGroup inline={false} label="field: ">
                <SegmentedControl small={true} inline={true}
                options={[
                    { label: 'PARAM', value: 'PARAM' },
                    { label: 'TAG', value: 'TAG' },]}
                onValueChange={this.handleCondTypeChange} value={cond_type}/>
                <HTMLSelect iconName='caret-down'
                 onChange={this.handleFieldChange} value={field_index}>
                    {/* <option value={-1} disabled hidden>field name</option> */}
                    <optgroup label="standard">
                        { cond_type == "PARAM" ?
                            fields.map((field, index) => (
                                field[1] != "json" && <option key={index} value={index}>{field[0]} | {field[1]}</option>))
                            :
                            tag_fields.map((field, index) => (
                                field[1] != "json" && <option key={index} value={index}>{field[0]} | {field[1]}</option>))
                        }
                    </optgroup>
                    <optgroup label="json">
                        { cond_type == "PARAM" ?
                            fields.map((field, index) => (
                                field[1] == "json" && <option key={index} value={index}>{field[0]} | {field[1]}</option>))
                            :
                            tag_fields.map((field, index) => (
                                field[1] == "json" && <option key={index} value={index}>{field[0]} | {field[1]}</option>))
                        }
                    </optgroup>
                </HTMLSelect>
                </FormGroup>
                {
                cond_type == "TAG" &&
                <Callout intent="warning">
                    Note: these conditions filter out results with no tags.
                    To filter out results with specific tags, use an overall negation ("None of these are true").
                </Callout>
                }
                {
                cond_type == "PARAM" && table_name == "epoch_group" &&
                <Callout intent="primary">
                    Note: epoch groups not defined by a single protocol are set to <Code>protocol_name="no_group_protocol"</Code>.
                </Callout>
                }
                { cond_type == "PARAM" && fields[field_index][1] == "json" &&
                <div>
                <FormGroup inline={false} label="Subfield: ">
                <InputGroup type="text" placeholder="json key" onValueChange={this.handleSubfieldChange} value={subfield_name} />
                <SegmentedControl small={true} inline={true}
                options={[
                    { label: 'string', value: 'string' },
                    { label: 'numeric', value: 'number' },]}
                onValueChange={this.handleSubfieldTypeChange} value={subfield_type}/>
                </FormGroup>
                </div>
                }
                { type_set && 
                <div>
                <ControlGroup>
                <HTMLSelect
                iconName='caret-down' onChange={this.handleOperatorChange} value={operator}>
                    <option value="" disabled hidden>Operator ...</option>
                {cur_operators &&
                cur_operators.map((operator, index) => (
                    <option key={index} value={operator[0]}>{operator[1]}</option>
                ))
                }
                </HTMLSelect>
                <InputGroup type="text" placeholder="value" fill={false} value={value}
                onValueChange={this.handleValueChange} />
                </ControlGroup>
                </div>
                }
            </div>
            :
            <p>Loading ... </p>}
            <ButtonGroup outlined={true}>
                <Button 
                small={true} icon="trash" intent='danger'
                onClick={this.props.onDelete}>delete</Button>
                <Button 
                small={true} icon="saved" intent='primary'
                disabled={!(this.state.operator && this.state.value)}
                onClick={this.sendUp}>save</Button>
            </ButtonGroup>
            {operator && value && unsaved_changes && 
            "Unsaved changes"}
        </div>
        {/* {isLoading ? <CircularProgress /> : null} */}
        </div>
    );
    }
}

export default CondBlock;
