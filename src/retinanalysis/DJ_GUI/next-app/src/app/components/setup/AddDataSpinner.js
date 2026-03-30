import React, { useEffect } from 'react';
import axios from 'axios';
import { CircularProgress } from '@mui/material';

export default function AddDataSpinner(props) {

    useEffect(() => {
        const interval = setInterval(() => {
            axios.get('http://localhost:3000/api/pop/is-adding')
                .then(response => {
                    if (!response.data.adding) {
                        props.onDoneLoading(true);
                    }
                })
                .catch(error => {
                    console.log(error.response.data.message);
                    props.onDoneLoading(true);
                });
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    return (
        <>
            <CircularProgress />
            <p>This can take a while. You can view progress in Terminal.</p>
        </>
    );
}