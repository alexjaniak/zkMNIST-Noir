import React from 'react';
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

const ProofDisplay: React.FC<ProofDisplayProps> = ({label, proof, prediction, onChain, offChain}) => {
    if (proof == undefined || prediction == undefined || label == undefined) return <div></div>;
    return (
        <div>
            <div>
                <h2>Expected Class:{label}</h2>
                <h2>Model Classification:{prediction}</h2>
            </div>
            <div>
                <h2>Verified Off-Chain: {offChain == undefined ? '-' : offChain.toString()}</h2>
                <h2>Verified On-Chain: {onChain == undefined ? '-' : onChain.toString()}</h2>
            </div>
            <h2>Proof</h2>
            <Uint8ArrayDisplay data={proof} />
        </div> 
    );
};

export default ProofDisplay;