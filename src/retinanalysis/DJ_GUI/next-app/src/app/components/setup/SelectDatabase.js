import React, { Component } from 'react';
import axios from 'axios';

import StartIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import ConnectIcon from '@mui/icons-material/Link';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import { IconButton, Alert, Snackbar, CircularProgress} from '@mui/material';

class SelectDatabase extends Component {
    constructor(props) {
    super(props);
    this.state = {
        data: null,
        error: null,
        response: null,
        newDatabaseName: '',
        open: false,
        isConnected: false,
        connected_db: '',
        started_db: '',
        isLoading: false,
    };
    }

    handleInputChange = (event) => {
    this.setState({ newDatabaseName: event.target.value });
    }

    fetchDatabases = () => {
    axios.get('http://localhost:3000/api/init/list-databases')
        .then(response => {
        // Handle success by setting the data in the state
        this.setState({ data: response.data.databases });
        })
        .catch(error => {
        // Handle error by setting the error message in the state
            this.setState({ error: error.response.data.message, response: null, open: true});
        });
    }

    componentDidMount() {
    this.fetchDatabases();
    }

    handleStart = (db) => {
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/init/start-database', { name: db })
            .then(response => {
            this.setState({ response: response.data.message, error: null, open: true});
            this.setState({ started_db: db });
            this.setState({ isLoading: false });
            })
            .catch(error => {
                this.setState({ error: error.response.data.message, response: null, open: true});
                this.setState({ isLoading: false });
            });
    }

    handleStop = (db) => {
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/init/stop-database', { name: db })
            .then(response => {
                this.setState({ response: response.data.message, error: null, open: true});
                if (this.state.isConnected && this.state.connected_db === db) {
                    this.setState({ isConnected: false , connected_db: '' });
                    this.props.onConnectionStatusChange(false);
                }
                if (this.state.started_db === db) {
                    this.setState({ started_db: '' });
                }
                this.setState({ isLoading: false });
            })
            .catch(error => {
                this.setState({ error: error.response.data.message, response: null, open: true});
                this.setState({ isLoading: false });
            });
    }

    handleConnect = (db) => {
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/init/connect-database', { name: db })
            .then(response => {
            this.setState({ response: response.data.message, error: null, open: true});
            this.setState({ isConnected: true, connected_db: db });
            this.props.onConnectionStatusChange(true);
            this.setState({ isLoading: false });
            })
            .catch(error => {
                this.setState({ error: error.response.data.message, response: null, open: true});
                this.setState({ isLoading: false });
            });
    }

    handleDelete = (db) => {
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/init/delete-database', { name: db })
            .then(response => {
            this.setState({ response: response.data.message, error: null, open: true});
            this.fetchDatabases();
            if (this.state.isConnected && this.state.connected_db === db) {
                this.setState({ isConnected: false , connected_db: '' });
                this.props.onConnectionStatusChange(false);
            }
            if (this.state.started_db === db) {
                this.setState({ started_db: '' });
            }
            this.setState({ isLoading: false });
            })
            .catch(error => {
                this.setState({ error: error.response.data.message, response: null, open: true});
                this.setState({ isLoading: false });
            });
    }

    handleCreate = () => {
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/init/create-database', { name: this.state.newDatabaseName })
            .then(response => {
            this.setState({ newDatabaseName: '', response: response.data.message, error: null, open: true});
            this.fetchDatabases();
            this.setState({ isLoading: false });
            })
            .catch(error => {
                this.setState({ error: error.response.data.message, response: null, open: true});
                this.setState({ isLoading: false });
            });
    }
    
    handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        this.setState({ open: false });
    }

    render() {
    const { data, error, response, newDatabaseName, open, connected_db, started_db, isLoading } = this.state;

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

        {data ? <div style={{ background: "darkgrey", borderRadius: "5%", padding: "2%" }}>
        <table><tbody>
        {data.map((db) => (
            <tr key={db}>
                <td>{db}</td>
                <td><IconButton onClick={() => {this.handleStart(db)}}><StartIcon /></IconButton></td>
                <td><IconButton onClick={() => {this.handleStop(db)}}><StopIcon /></IconButton></td>
                <td><IconButton onClick={() => {this.handleConnect(db)}}><ConnectIcon /></IconButton></td>
                <td><IconButton onClick={() => {this.handleDelete(db)}}><DeleteIcon /></IconButton></td>
            </tr>
        ))}
        </tbody></table>
        <input type="text" value={newDatabaseName} onChange={this.handleInputChange} placeholder='create new ...' />
        <IconButton onClick={this.handleCreate}><AddIcon /></IconButton>
        {isLoading && (
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                    <CircularProgress />
                </div>
            )}
        {started_db != '' && <p>Started database: {started_db}</p>}
        {connected_db != '' && <p>Connected database: {connected_db}</p>}
    </div> : <p>Loading...</p>}
        </div>
    );
    }
}

export default SelectDatabase;
