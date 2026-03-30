"use client"

// BlueprintJS CSS
import "normalize.css";
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";

import styles from './page.module.css';
import './globals.css';
import * as React from 'react';

import ResultsViewer from './components/ResultsViewer';
import SetUpStepper from './components/SetUpStepper';

export default function Home() {
  const [results, setResults] = React.useState(null);

  const handleDisplayResults = (results) => {
    setResults(results);
  }

  return (
    <main>
      <div style={{ display: 'flex', height: '100vh'}}>
        <div style={{ flex: 1, overflow: 'auto'}} className={styles.card}>
          <SetUpStepper onResultsChange={handleDisplayResults}/>
        </div>
        <div style={{ flex: 2}} className={styles.card}>
          <ResultsViewer results={results} />
        </div>
      </div>
    </main>
  );
}
