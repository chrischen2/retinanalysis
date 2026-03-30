"use client"

import * as React from 'react';
import axios from 'axios';

import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import StepContent from '@mui/material/StepContent';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';

import SelectDatabase from './setup/SelectDatabase';
import SetUser from './setup/SetUser';
import AddData from './setup/AddData';
import QueryContainer from './setup/QueryContainer';

const steps = [
  {
    label: 'Select database',
    description: `Current list of databases: start and connect to continue.`,
  },
  {
    label: 'Set user',
    description:
      'Set a user to continue. If a user is already set, you can skip this step.',
  },
  {
    label: 'Add data (optional)',
    description: ``,
  },
  {
    label: 'Query data',
    description: `...`,
  },
];

export default function SetUpStepper(props){
  const [activeStep, setActiveStep] = React.useState(0);
  const [isConnected, setIsConnected] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [user, setUser] = React.useState(false);
  const [cont, setCont] = React.useState(false);
  const [queryObj, setQueryObj] = React.useState(null);
  const [excludeLevels, setExcludeLevels] = React.useState([]);
  const [response, setResponse] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [open, setOpen] = React.useState(false);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep === 3 ? prevActiveStep : prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
  };

  const handleConnection = (status) => {
    setIsConnected(status);
  }

  const handleUser = (status) => {
    setUser(status);
  }

  const handleCont = (status) => {
    setCont(status);
  }

  const handleExcludeChange = (levels) => {
    setExcludeLevels(levels);
  }

  const handleQueryObj = (obj) => {
    console.log(obj);
    if (obj == {}) {
      setQueryObj(null);
    } else {
      setQueryObj(obj);
    }
  }

  const handleExec = () => {
    setIsLoading(true);
    axios.post('http://localhost:3000/api/query/execute-query', { 
      query_obj: queryObj,
      exclude_levels: excludeLevels
     })
        .then(response => {
          setIsLoading(false);
          if (response.data.results) {
            props.onResultsChange(response.data.results);
          } else {
            setResponse(response.data.message);
            setError(null);
            setOpen(true);
          }
        })
        .catch(error => {
            setIsLoading(false);
            setError(error.response.data.message);
            setResponse(null);
            setOpen(true);
        });
  }

  const handleClose = (event, reason) => {
    if (reason === 'clickaway') {
        return;
    }
    setOpen(false);
  }

  return (
    <Box sx={{ maxWidth: "none" }}>
      <Stepper activeStep={activeStep} orientation="vertical" sx={{
        '& .MuiStepConnector-line': {minHeight: "2px"},
      }}>
        {steps.map((step, index) => (
          <Step key={step.label}>
            <StepLabel optional={index === 3 ? (<Typography variant="caption">Last step</Typography>) : null}>
              {step.label}
            </StepLabel>
            <StepContent>
              {step.description}
              {index === 0 && <SelectDatabase onConnectionStatusChange={handleConnection} />}
              {index === 1 && <SetUser onUserSet={handleUser} />}
              {index === 2 && <AddData onContinue={handleCont} />}
              {index === 3 && <QueryContainer 
                                onQueryObj={handleQueryObj} 
                                onExcludeChange={handleExcludeChange}/>}
              {isLoading ? 
              <CircularProgress />
              :
              <Box sx={{ mb: 2 }}>
                <div>
                  <Button
                    disabled={(activeStep === 0 && !isConnected) || (activeStep === 1 && !user) 
                      || (activeStep === 2 && !cont) || (activeStep === 3 && !queryObj)}
                    variant="contained"
                    onClick={index === steps.length - 1 ? handleExec : handleNext}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    {index === steps.length - 1 ? 'View Results' : 'Next'}
                  </Button>
                  <Button
                    disabled={index === 0}
                    onClick={handleBack}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    Back
                  </Button>
                </div>
              </Box>}
            </StepContent>
          </Step>
        ))}
      </Stepper>
      {activeStep === steps.length && (
        <Paper square elevation={0} sx={{ p: 3 }}>
          <Typography></Typography>
          <Button onClick={handleReset} sx={{ mt: 1, mr: 1 }}>
            Reset
          </Button>
        </Paper>
      )}
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
    </Box>
  );
}
