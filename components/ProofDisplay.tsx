import React from 'react';
//@ts-ignore
import styles from './ProofDisplay.module.css';


type ProofDisplayProps = {
    label: number | undefined,
    prediction: number | undefined,
    proof:  Uint8Array | undefined,
    offChain: boolean | undefined,
    onChain: boolean | undefined
}

const Uint8ArrayDisplay = ({ data }) => {
    // Convert each byte to its hexadecimal representation and join them with a space
    const hexString = Array.from(data)
      .map((byte: any) => byte.toString(16).padStart(2, '0'))
      .join(' ');
  
    return <div className={styles.arrayDisplay}>{hexString.toUpperCase()}</div>;
  };

function handleVerification(output: boolean | undefined) {
    if (output == undefined) return (<span style={{ display: 'inline-block', width: '50px', textAlign: 'center'}}>—</span>);

    return (<span style={{ display: 'inline-block', width: '50px', textAlign: 'center', color: output ? 'green' : 'red' }}>
                {output ? 'True' : 'False'}
            </span>);
}

const ProofDisplay: React.FC<ProofDisplayProps> = ({label, proof, prediction, onChain, offChain}) => {
    if (proof == undefined || prediction == undefined || label == undefined) return <div></div>;
    return (
        <div>
            <hr></hr>
            <div className={styles.outputContainer}>
                <h2 className={styles.outputHeader}>Expected Class: {label}</h2>
                <h2 className={styles.outputHeader}>
                    Verified On-Chain: {handleVerification(onChain)}
                </h2>
            </div>
            <div className={styles.outputContainer}>
                <h2 className={styles.outputHeader}>Model Classification: {prediction}</h2>
                <h2 className={styles.outputHeader}>Verified Off-Chain: {handleVerification(offChain)}</h2>
            </div>
            <h2 className={styles.proofHeader}>Proof</h2>
            <Uint8ArrayDisplay data={proof} />
        </div> 
    );
};

export default ProofDisplay;