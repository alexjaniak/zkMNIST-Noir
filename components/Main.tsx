import { useState, useEffect } from 'react';

import { toast } from 'react-toastify';
import Ethers from '../utils/ethers';
import React from 'react';

import { Noir } from '@noir-lang/noir_js';
import { BarretenbergBackend } from '@noir-lang/backend_barretenberg';
import { CompiledCircuit, ProofData } from '@noir-lang/types';
import newCompiler, { compile } from '@noir-lang/noir_wasm';
import { initializeResolver } from '@noir-lang/source-resolver';
import axios from 'axios';

import { forwardPass } from '../utils/ml/model';
import * as tf from "@tensorflow/tfjs";
import DigitImage from './DigitImage';
import ProofDisplay from './ProofDisplay';

import styles from './Main.module.css';

// Get compiled circuit
async function getCircuit(name: string) {
  await newCompiler();
  const { data: noirSource } = await axios.get('/api/readCircuitFile?filename=' + name);

  initializeResolver((id: string) => {
    const source = noirSource;
    return source;
  });

  const compiled = compile('main');
  return compiled;
}

function MainComponent(sampleData) {
  const [proof, setProof] = useState<ProofData>();
  const [noir, setNoir] = useState<Noir | null>(null);
  const [model, setModel] = useState(null);
  const [backend, setBackend] = useState<BarretenbergBackend | null>(null);

  const [selectedDigit, setSelectedDigit] = useState<string | undefined>(undefined);
  const [prediction, setPrediction] = useState<number | undefined>(undefined);
  const [provedDigit, setProvedDigit] = useState<number | undefined>(undefined);
  const [offChainVerification, setOffChainVerification] = useState<boolean | undefined>();
  const [onChainVerification, setOnChainVerification] = useState<boolean | undefined>();
  
  // Classify and Prove Digit
  const calculateProof = async () => {
    const calc = new Promise(async (resolve, reject) => {
      if (noir && model && selectedDigit) {
        // @ts-ignore
        const input = tf.tensor(sampleData[selectedDigit]).reshape([1, 28, 28]).div(tf.scalar(255.0));
        setProvedDigit(+selectedDigit);

        // @ts-ignore
        const [scaledInput, scaledWeights, scaledBias, output] = forwardPass(model, input);
        
        /* DEBUG OUTPUT
        // @ts-ignore
        console.log("Expected Output:", +selectedDigit);
        output.data().then(arr => console.log("Model Inference: ", arr));
        */ 

        const outputClass = (await output.array())[0];
        setPrediction(outputClass);

        const flattenedScaledWeights = scaledWeights.flatten();
        const abi = {
            input: (await scaledInput.array())[0],
            weights: await flattenedScaledWeights.array(),
            biases: await scaledBias.array(),
            class: outputClass,
        };

        const { proof, publicInputs } = await noir!.generateFinalProof(abi);

        //console.log('Proof created: ', proof);
        
        setProof({ proof, publicInputs });
        setOffChainVerification(undefined);
        setOnChainVerification(undefined);
        resolve(proof);
      } else reject(new Error("Model or Noir not initialized"));
    });
      

    toast.promise(calc, {
      pending: 'Calculating proof...',
      success: 'Proof calculated!',
      error: 'Error calculating proof',
    });
  };

  // Handle Off-Chain Verification
  const offChainHandler = async () => {
    const verifyOffChain = new Promise(async (resolve, reject) => {
      if (!proof) return reject(new Error('No proof'));

      const verification = await noir!.verifyFinalProof({
        proof: proof.proof,
        publicInputs: proof.publicInputs,
      });
      console.log('Proof verified off-chain: ', verification);
      setOffChainVerification(verification);
      resolve(verification);
    });

    toast.promise(verifyOffChain, {
      pending: 'Verifying proof off-chain...',
      success: 'Proof verified off-chain!',
      error: 'Error verifying proof',
    });
  }

  // Handle On-Chain Verification 
  const onChainHandler = async () => {
    const verifyOnChain = new Promise(async (resolve, reject) => {
      if (!proof) return reject(new Error('No proof'));
      if (!window.ethereum) return reject(new Error('No ethereum provider'));
      try {
        const ethers = new Ethers();
        const verification = await ethers.contract.verify(proof.proof, proof.publicInputs);

        console.log('Proof verified on-chain: ', verification);
        setOnChainVerification(verification);
        resolve(verification);
      } catch (err) {
        console.log(err);
        reject(new Error("Couldn't verify proof on-chain"));
      }
    });

    toast.promise(verifyOnChain, {
      pending: 'Verifying proof on-chain...',
      success: 'Proof verified on-chain!',
      error: {
        render({ data }: any) {
          return `Error: ${data.message}`;
        },
      },
    });
  }

  const initNoir = async () => {
    const circuit = await getCircuit('main');

    const backend = new BarretenbergBackend(circuit as CompiledCircuit, { threads: 8 });
    setBackend(backend);
    const noir = new Noir(circuit as CompiledCircuit, backend);

    await toast.promise(noir.init(), {
      pending: 'Initializing Noir...',
      success: 'Noir initialized!',
      error: 'Error initializing Noir',
    });

    setNoir(noir);
  };

  const loadModel = async () => {
    const model = tf.loadLayersModel("http://localhost:3000/api/readModel/model.json");

    await toast.promise(model, {
      pending: 'Loading Model...',
      success: 'Model loaded!',
      error: 'Error loading Model',
    });

    // @ts-ignore
    setModel(await model);
  };   

  // Initialize Noir & load Model at the beginning of render
  useEffect(() => {
    initNoir(); // init Noir
    loadModel(); // load Model
  }, []);

  // Array of MNIST labels
  const MNISTLabels: string[] = Array.from({ length: 10 }, (_, i) => i.toString());

  return (
    <div className={styles.container}>
      <h1 style={{textAlign: 'center'}} >ZKMNIST - Noir</h1>
      <a href={'https://github.com/alexjaniak/zkMNIST-Noir'}>@GitHub</a>
      <hr></hr>
      <p>
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla at risus mauris. Cras ullamcorper vestibulum nibh, sed efficitur diam accumsan eget. Sed sit amet ante id orci tincidunt hendrerit vel vel arcu. Nam dictum lectus nec felis auctor, at eleifend purus semper. Aliquam at vestibulum libero, consectetur hendrerit tellus. Maecenas nec nisl nibh. Vivamus tempor quam at lacus viverra, lobortis pretium tellus rhoncus. Curabitur in porta nisi. Nam ultrices dictum commodo. Morbi vel sollicitudin urna. Nam suscipit faucibus metus, eget ornare elit tempor nec. Proin a elit a enim iaculis mollis quis imperdiet dui. Sed viverra mauris et velit venenatis ornare. Vivamus erat nibh, venenatis eget auctor nec, cursus nec mauris. Nunc pulvinar magna sed odio vulputate lobortis. Integer sodales orci nec tempor semper. Suspendisse condimentum ultrices justo at lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Vivamus vitae massa sit amet augue malesuada mattis. Duis interdum, elit sed pretium commodo, tellus nulla euismod nulla, sit amet feugiat turpis nisl non lacus. Cras ante erat, suscipit ut turpis eget, ornare vulputate urna. Sed eget commodo ante. Ut fermentum nisl et risus mattis placerat. Praesent malesuada mauris et eros blandit eleifend. Donec id convallis augue. Donec efficitur metus quis suscipit vulputate. Suspendisse gravida felis turpis, ut vestibulum risus fringilla eu. Vestibulum venenatis leo ac maximus hendrerit. Ut maximus tincidunt est, ac dapibus nulla accumsan in. Donec tempor justo porta ipsum blandit malesuada. Maecenas ultricies libero ut porttitor porta. Donec fermentum aliquam lacus non vestibulum. Sed.
      </p>
      <h2>Select Digit</h2>
      <div className={styles.digitGrid}>
        {MNISTLabels.map(label => (
          <div key={+label}> 
            <DigitImage 
              label={+label} 
              data={sampleData[label]}
              scale={4}
              onClick={() => setSelectedDigit(label)}
              isSelected={label == selectedDigit}
            />
          </div>
        ))}
      </div>
      <div>
        <button onClick={calculateProof}>Classify & Prove</button>
        <button onClick={offChainHandler}>Verify Off-Chain</button>
        <button onClick={onChainHandler}>Verify On-Chain</button>
      </div>
      <ProofDisplay
        label={provedDigit}
        prediction={prediction} 
        proof={proof ? proof?.proof : undefined}
        offChain={offChainVerification}
        onChain={onChainVerification}
      />
    </div>
  );
}

export default MainComponent;
