import React, { Component } from 'react';
import axios from 'axios';

import DoneIcon from '@mui/icons-material/Done';
import { IconButton, Alert, Snackbar, CircularProgress} from '@mui/material';

class SetUser extends Component {
    constructor(props) {
    super(props);
    this.state = {
        user: null,
        open: false,
        error: null,
        response: null,
        newUsername: '',
        isConnected: false,
        isLoading: false,
    };
    }

    handleInputChange = (event) => {
    this.setState({ newUsername: event.target.value });
    }

    fetchUser = () => {
    axios.get('http://localhost:3000/api/user/get-user')
        .then(response => {
        // Handle success by setting the data in the state
        this.setState({ user: response.data.user });
        if(response.data.user != "user_not_set"){
            this.props.onUserSet(true);
        }
        })
        .catch(error => {
        // Handle error by setting the error message in the state
            this.setState({ error: error.response.data.message, response: null, open: true});
        });
    }

    componentDidMount() {
    this.fetchUser();
    }

    handleUser = () => {
        this.setState({ isLoading: true });
        axios.post('http://localhost:3000/api/user/set-user', { user: this.state.newUsername })
            .then(response => {
            this.setState({ newUsername: '', response: response.data.message, error: null, open: true});
            this.fetchUser();
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
    const { user, error, response, newUsername, open, isLoading } = this.state;

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

        {user ? <div style={{ background: "darkgrey", borderRadius: "5%", padding: "2%" }}>
        <input type="text" value={newUsername} onChange={this.handleInputChange} placeholder='set username...' />
        <IconButton onClick={this.handleUser}><DoneIcon /></IconButton>
        {isLoading && (
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                    <p>This can take a while, depending on how many files you have. You can check the terminal to view progress.</p>
                    <CircularProgress />
                </div>
            )}
        {user != "user_not_set" ? <p>Current user: {user}</p> : <p>No user set</p>}
    </div> : <p>Loading...</p>}
        </div>
    );
    }
}

export default SetUser;
