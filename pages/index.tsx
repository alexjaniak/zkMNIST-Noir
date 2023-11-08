import React from 'react';
import MainComponent from '../components/Main';
import fs from 'fs/promises';
import path from 'path';
import axios from 'axios';

export async function getServerSideProps() {
  const dataFilePath = path.join(process.cwd(), 'utils/ml/data', 'sample.json');
  const jsonData = await fs.readFile(dataFilePath, 'utf8');
  const sampleData = JSON.parse(jsonData);
  
  return {
    props: {
      sampleData
    },
  };
}
export default function Page({ sampleData }) {
  return <MainComponent {...sampleData}/>;
}