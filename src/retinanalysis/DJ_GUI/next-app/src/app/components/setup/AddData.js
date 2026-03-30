import React, { Component } from 'react';
import axios from 'axios';

import { Alert, Snackbar, CircularProgress} from '@mui/material';
import { ButtonGroup, Button } from '@blueprintjs/core';

import AddDataSpinner from './AddDataSpinner';

class AddData extends Component {
    constructor(props) {
    super(props);
    this.state = {
        empty: null,
        num_experiments: null,
        formData: {data_dir: '',
            meta_dir: '',
            tags_dir: ''},
        error: null,
        open: false,
        response: null,
        isLoading: false,
        isRunning: false,
    };
    }

    isEmpty = () => {
    axios.get('http://localhost:3000/api/pop/is-empty')
        .then(response => {
        // Handle success by setting the data in the state
        this.setState({ empty: response.data.empty, num_experiments: response.data.num_experiments });
        this.props.onContinue(!response.data.empty);
        })
        .catch(error => {
        // Handle error by setting the error message in the state
            this.setState({ error: error.response.data.message, response: null, open: true});
        });
    }

    componentDidMount() {
        this.isEmpty();
    }

    handleChange = (event) => {
        const { name, value } = event.target;
        this.setState({ formData: {
            ...this.state.formData,
            [name]: value
        }
        });
    };

    handleSubmit = (event) => {
        event.preventDefault();
        console.log('Form Data:', this.state.formData);
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/pop/add-data', 
            { data_dir: this.state.formData.data_dir, meta_dir: this.state.formData.meta_dir, tags_dir: this.state.formData.tags_dir })
            .then(response => {
            this.setState({ isRunning: true, isLoading: false });
            this.setState({ response: response.data.message, error: null, open: true});
            })
            .catch(error => {
                this.setState({ error: error.response.data.message, response: null, open: true});
                this.isEmpty();
                this.setState({ isLoading: false });
            });
    };

    handleAddedData = (status) => {
        if (status) {
            this.isEmpty();
            this.setState({ isLoading: false, isRunning: false });
            this.setState({ response: "Data added successfully!", error: null, open: true});
        }
    }

    handleClear = (event) => {
        event.preventDefault();
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/pop/clear')
            .then(response => {
            this.setState({ response: response.data.message, error: null, open: true});
            this.isEmpty();
            this.setState({ isLoading: false });
            })
            .catch(error => {
                this.setState({ error: error.response.data.message, response: null, open: true});
                this.isEmpty();
                this.setState({ isLoading: false });
            });
    };
    
    handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        this.setState({ open: false });
    }

    render() {
    const { empty, num_experiments, formData, error, response, open, isLoading, isRunning } = this.state;
    const inputFields = [
        { id: 'data_dir', label: 'Data Directory' },
        { id: 'meta_dir', label: 'Meta Directory' },
        { id: 'tags_dir', label: 'Tags Directory' }
    ];

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

        {empty != null ? <div style={{ background: "darkgrey", borderRadius: "5%", padding: "2%" }}>
        {empty ? <p>There are no experiments in the database. Add data to continue.</p> 
        : <p>There are {num_experiments} experiments in the database. You can add more data to the database below.</p>}
        <form onSubmit={this.handleSubmit}>
            {inputFields.map((field) => (
                <div key={field.id}>
                    <input
                        type="text"
                        id={field.id}
                        name={field.id}
                        placeholder={`Enter ${field.label} ...`}
                        value={formData[field.id]}
                        onChange={this.handleChange}
                    />
                </div>
            ))}
            <ButtonGroup outlined={true}>
                <Button small={true} icon="add" intent='success'
                type="submit">Add Data</Button>
                <Button small={true} icon="trash" intent='danger'
                onClick={this.handleClear}>Clear Database</Button>
            </ButtonGroup>
        </form>
        {isLoading ? <CircularProgress /> : null}
        {isRunning ? <AddDataSpinner onDoneLoading={this.handleAddedData} /> : null}
    </div> : <p>Loading...</p>}
        </div>
    );
    }
}

export default AddData;
