import React, { Component } from 'react';
import axios from 'axios';

import { CircularProgress } from '@mui/material';

class Information extends Component {
    constructor(props) {
    super(props);
    this.state = {
        options: null,
        cur_option: null,
        img: null,
        selected: null,
        id: null,
        isLoading: false,
        message: null
    };
    }

    fetchImage = (cur_option) => {
    this.setState({ isLoading: true });
    axios.post('http://localhost:3000/api/results/get-visualization', {
        data: cur_option
    })
        .then(response => {
        // Handle success by setting the data in the state
        this.setState({ isLoading: false });
        if (response.data.image) {
            this.setState({ img: response.data.image });
            this.setState({ message: null });
        } else {
            this.setState({ message: response.data.message });
        }
        })
        .catch(error => {
        this.setState({ isLoading: false });
        // Handle error by sending error message to console
        console.log(error.response.data.message);
        });
    }



    fetchOptions = () => {
    this.setState({ isLoading: true });
    axios.post('http://localhost:3000/api/results/get-visualization-data', {
        level: this.props.level,
        id: this.props.id,
        experiment_id: this.props.experiment_id
    })
        .then(response => {
        // Handle success by setting the data in the state
        this.setState({ isLoading: false });
        if (response.data.options != null) {
            this.setState({ options: response.data.options });
            let optgroup = null
            let index = null;
            if (this.state.selected == null 
                || !(this.state.selected.split('-')[0] in response.data.options)
                || parseInt(this.state.selected.split('-')[1]) >= response.data.options[
                    (this.state.selected.split('-')[0])].length) {
                optgroup = Object.keys(response.data.options)[0];
                index = 0;
                this.setState({ selected: optgroup + '-' + index });
            } else {
                optgroup = this.state.selected.split('-')[0];
                index = parseInt(this.state.selected.split('-')[1]);
            }
            this.setState({ cur_option: response.data.options[optgroup][index] });
            this.fetchImage(response.data.options[optgroup][index]);
        } else {
            this.setState({ message: response.data.message });
        }
        })
        .catch(error => {
            this.setState({ isLoading: false });
            // Handle error by sending error message to console
            console.log(error.response.data.message);
        });
    }

    componentDidMount() {
        if (this.props.id){
            this.fetchOptions();
        } else {
            this.setState({ options: null });
        }
    }

    handleOptionChange = (event) => {
        this.setState({ selected: event.target.value });
        let optgroup = event.target.value.split('-')[0];
        let index = parseInt(event.target.value.split('-')[1]);
        // console.log("option change: " + optgroup + " " + index);
        this.setState({ cur_option: this.state.options[optgroup][index] });
        this.fetchImage(this.state.options[optgroup][index]);
    }

    componentDidUpdate(prevProps) {
    if (prevProps !== this.props) {
        if (this.props.id){
            this.fetchOptions();
        } else {
            this.setState({ options: null });
        }
    }
    }

    render() {
    const { img, options, selected} = this.state;

    return (
        <div>
        {options &&
        <div>
            Select device ({this.props.level} {this.props.id}):
            <select onChange={this.handleOptionChange} value={selected}>
            {Object.keys(options).map((key) => (
                <optgroup key={key} label={key}>
                    {options[key].map((option, index) => (
                        <option key={`${key}-${index}`} value={`${key}-${index}`}>
                            {option.label}
                        </option>
                    ))}
                </optgroup>
            ))}
            {/* <optgroup label='Responses'>
            {options.responses.map((option, index) => (
                <option key={"responses-" + index} 
                        value={"responses-" + index}>{option.device_name}</option>
            ))}
            </optgroup>
            <optgroup label='Stimuli'>
            {options.stimuli.map((option, index) => (
                <option key={"stimuli-" + index} 
                        value={"stimuli-" + index}>{option.device_name}</option>
            ))}
            </optgroup> */}
            </select>
        </div>}
        {!this.state.isLoading && img && 
        <img style = {{ objectFit: 'contain', width: '100%', height: '100%' }}
        src={'data:image/png;base64,' + img} />}
        {this.state.isLoading && <><CircularProgress /><p>Loading from server ...</p></>}
        </div>
    );
    }
}

export default Information;
