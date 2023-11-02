import React from 'react';
import Component from '../components/component';
import fs from 'fs/promises';
import path from 'path';

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
  return <Component {...sampleData}/>;
}