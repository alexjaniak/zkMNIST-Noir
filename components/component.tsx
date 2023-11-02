import { useState, useEffect, SetStateAction } from 'react';

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

function Component(sampleData) {
  const [proof, setProof] = useState<ProofData>();
  const [noir, setNoir] = useState<Noir | null>(null);
  const [model, setModel] = useState(null);
  const [backend, setBackend] = useState<BarretenbergBackend | null>(null);


  // Calculates proof
  const calculateProof = async () => {
    const calc = new Promise(async (resolve, reject) => {
      if (noir && model) {
        const input = tf.tensor(sampleData['1']).reshape([1, 28, 28]).div(tf.scalar(255.0));
        console.log("Expected Output:", 1);
        // @ts-ignore
        const [scaledInput, scaledWeights, scaledBias, output] = forwardPass(model, input);

        output.data().then(arr => console.log("Model Inference: ", arr));

        const flattenedScaledWeights = scaledWeights.flatten();
        const abi = {
            input: (await scaledInput.array())[0],
            weights: await flattenedScaledWeights.array(),
            biases: await scaledBias.array(),
            class: (await output.array())[0],
        };

        const { proof, publicInputs } = await noir!.generateFinalProof(abi);
        console.log('Proof created: ', proof);
        setProof({ proof, publicInputs });
        resolve(proof);
      } else reject(new Error("Model or Noir not initialized"));
    });
      

    toast.promise(calc, {
      pending: 'Calculating proof...',
      success: 'Proof calculated!',
      error: 'Error calculating proof',
    });
  };

  const verifyProof = async () => {
    const verifyOffChain = new Promise(async (resolve, reject) => {
      if (proof) {
        const verification = await noir!.verifyFinalProof({
          proof: proof.proof,
          publicInputs: proof.publicInputs,
        });

        console.log('Proof verified off-chain: ', verification);
        resolve(verification);
      }
    });

    const verifyOnChain = new Promise(async (resolve, reject) => {
      if (!proof) return reject(new Error('No proof'));
      if (!window.ethereum) return reject(new Error('No ethereum provider'));
      try {
        const ethers = new Ethers();
        const verification = await ethers.contract.verify(proof.proof, proof.publicInputs);

        console.log('Proof verified on-chain: ', verification);
        resolve(verification);
      } catch (err) {
        console.log(err);
        reject(new Error("Couldn't verify proof on-chain"));
      }
    });

    toast.promise(verifyOffChain, {
      pending: 'Verifying proof off-chain...',
      success: 'Proof verified off-chain!',
      error: 'Error verifying proof',
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
  };

  // Verifier the proof if there's one in state
  useEffect(() => {
    if (proof) {
      verifyProof();

      return () => {
        // TODO: Backend should be destroyed by Noir JS so we don't have to
        // store backend in state
        //backend!.destroy(); // ? This breaks off-chain verification ? 
      };
    }
  }, [proof]);

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

  useEffect(() => {
    initNoir();
  }, []);

  // Load model at the beginning of render
  useEffect(() => {
    const loadModel = async () => {
      // Assume a model located at '/path/to/model.json'
      const model = await tf.loadLayersModel("http://localhost:3000/api/readModel/model.json");
      // @ts-ignore
      setModel(model);
    };   
    loadModel();
  }, []);

  return (
    <div className="container">
      <h1>ZKMNIST - Noir</h1>
      <button onClick={calculateProof}>Calculate proof</button>
    </div>
  );
}

export default Component;
