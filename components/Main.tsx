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

//@ts-ignore
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
      <a style={{fontSize: '20px'}} href={'https://github.com/alexjaniak/zkMNIST-Noir'}>@GitHub</a>
      <hr></hr>
      <p>
        Ethereum's smart contracts enable trustless execution in a decentralized environment. Yet, the limitations in computational power and the transparent aspect of blockchain transactions limit the ability to develop applications that require intensive computation or handle private data, like those used in machine learning. Zero-Knowledge (ZK) proofs aim to free these constraints by allowing the validation of transactions or computations without revealing the underlying data, thereby enabling privacy-preserving smart contracts. 
      </p>
      <p>
        These proofs can thus be used to facilitate the on-chain execution of machine learning models in a way that keeps the data used for inference confidential. This web app is an example of such zero-knowledge machine learning with the proving system written using Aztec Network’s ZK domain specific language <a href='https://noir-lang.org/'>Noir</a>. The app uses a neural network trained on the <a href='https://paperswithcode.com/dataset/mnist'>MNIST dataset</a> (A collection of handwritten digits), that classifies the selected digit and then generates proof of the classification while keeping the inputs private. The proof can then be verified on or off-chain. 
      </p>
      <p>
        Inspired by <a href={'https://0xparc.org/blog/zk-mnist'}>0xPARC</a>. Authored by <a href={'https://alexjaniak.com'}>Alexander Janiak</a>.
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
