import React from 'react';
import Link from 'next/link';

//@ts-ignore
import styles from '../components/Main.module.css';

import Latex from 'react-latex-next';
import 'katex/dist/katex.min.css';

const ModelTable = () => {
    const tableStyle: React.CSSProperties= {
      borderCollapse: 'collapse',
      width: '100%',
      marginTop: '20px',
    };
  
    const thStyle: React.CSSProperties = {
      border: '1px solid #dddddd',
      textAlign: 'left',
      padding: '8px',
      backgroundColor: "#f2f2f2",
    };
  
    const tdStyle: React.CSSProperties = {
      border: '1px solid #dddddd',
      textAlign: 'left',
      padding: '8px',
    };
  
    return (
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={thStyle}>Layer (type)</th>
            <th style={thStyle}>Output Shape</th>
            <th style={thStyle}>Param #</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={tdStyle}>flatten (Flatten)</td>
            <td style={tdStyle}>(None, 784)</td>
            <td style={tdStyle}>0</td>
          </tr>
          <tr>
            <td style={tdStyle}>dense (Dense)</td>
            <td style={tdStyle}>(None, 300)</td>
            <td style={tdStyle}>235,500</td>
          </tr>
          <tr>
            <td style={tdStyle}>dense_1 (Dense)</td>
            <td style={tdStyle}>(None, 100)</td>
            <td style={tdStyle}>30,100</td>
          </tr>
          <tr>
            <td style={tdStyle}>dense_2 (Dense)</td>
            <td style={tdStyle}>(None, 30)</td>
            <td style={tdStyle}>3,030</td>
          </tr>
          <tr>
            <td style={tdStyle}>dense_3 (Dense)</td>
            <td style={tdStyle}>(None, 10)</td>
            <td style={tdStyle}>310</td>
          </tr>
          <tr>
            <td style={tdStyle}>Total params: 268,940 (1.03 MB)</td>
            <td style={tdStyle}>Trainable params: 268,940 (1.03 MB)</td>
            <td style={tdStyle}>Non-trainable params: 0 (0.00 Byte)</td>
          </tr>
        </tbody>
      </table>
    );
  };
  
export default function Page() {
    return (
        <div className={styles.container}>
            <h1 style={{textAlign: 'center'}} >ZKMNIST - Noir</h1>
            <div>
              <a style={{fontSize: '20px'}} href={'https://github.com/alexjaniak/zkMNIST-Noir'}>@GitHub</a>
              <span style={{marginLeft: '10px', marginRight: '10px'}}>|</span>
              <Link style={{fontSize: '20px'}} href='/'>Demo</Link>
            </div>
            <hr></hr>
            <h2>About</h2>
            <p><a href="https://ethereum.org/en/">Ethereum's</a> smart contracts enable trustless execution in a decentralized environment. Yet, the limitations in computational power and the transparent aspect of blockchain transactions limit the ability to develop applications that require intensive computation or handle private data, like those used in machine learning. Zero-Knowledge (ZK) proofs aim to free these constraints by allowing off-chain computations to be verified on-chain. Furthermore, they allow the verification of these computations without revealing the underlying data, thereby enabling privacy-preserving smart contracts.</p>
            <p>Similarly, these proofs can be used to facilitate the execution of machine learning models in a way that keeps the input data or the model weights confidential. This might be useful when input data is sensitive and private, such as personal biometric information. Likewise, model parameters can be confidential, like those used in biometric authentication systems.</p>
            <p>At the same time, it might be essential for downstream entities, such as on-chain smart contracts, that rely on the output of the machine learning model, to be certain that the input was correctly processed by the ML model to yield the claimed output. This ensures both privacy and reliability in the application of these models in sensitive areas.</p>
            <p>This web app is an example of such zero knowledge machine learning. The app uses a neural network trained on the <a href="https://paperswithcode.com/dataset/mnist">MNIST dataset</a> (A collection of handwritten digits), that classifies a selected digit and then generates proof of the classification while keeping the inputs (the pixels) private. The proof can then be verified on or off-chain.</p>
            <p>The ZK proving system is written using <a href="https://aztec.network/">Aztec Network’s</a> ZK domain specific language <a href="https://noir-lang.org/">Noir</a>. The web application was built from Aztec's <a href="https://github.com/noir-lang/noir-starter/tree/main/next-hardhat">Noir Starter</a>. It uses <a href="https://nextjs.org/">Next.js</a> as the frontend framework, <a href="https://hardhat.org/">Hardhat</a> to deploy and test, and <a href="https://www.tensorflow.org/">TensorFlow/Keras</a> for model training and prediction.</p>
            <p>Inspired by <a href="https://0xparc.org/blog/zk-mnist">0xPARC's ZK Machine Learning</a>. Authored by <a href="https://alexjaniak.com">Alexander Janiak.</a></p>
            <hr></hr>
            <h2>Model & Data</h2>
            <p>The MNIST dataset is a collection of 70,000 handwritten digits ranging from 0 to 9, commonly used for training and testing in the field of machine learning. It was created by merging samples from a dataset of American Census Bureau employees and high school students to ensure a diverse set of handwriting styles. Each image is grayscale, 28x28 pixels, and labeled with the corresponding digit it represents. MNIST was designed to be a benchmark dataset to evaluate the performance of algorithms in accurately recognizing and classifying handwritten digits. It has become a standard for evaluating machine learning techniques and therefore a perfect dataset to test Noir's capabilities for zero knowledge machine learning.</p>
            <div className={styles.imageContainer}>
                <img 
                    src="digits.png"
                    alt="Example of MNIST Digits"
                />
            </div>
            <p>Unlike 0xPARC's CNN approach, the chosen model architecture is a simple dense network:</p>
            <ModelTable></ModelTable>
            <p>The model uses the ReLu activation function between dense layers and a Softmax function after the last layer. The model was trained using Stochastic Gradient Descent and uses Sparse Categorical Cross Entropy for the loss function. After training, the model achieved 97.52% accuracy on the test set.</p>
            <hr></hr>
            <h2>Approach</h2>
            <p>The approach used was similar to 0xPARC's implementation with a few alterations to account for changes in the model and Noir's quirks. Like in 0xPARC's demo, and to simplify the circuit, only the last layer of the model was implemented as a zk-SNARK. Noir's simple Rust-like syntax made writing the circuit easy.</p>
            <p>The computation for the neural network's last layer forward pass implemented as a circuit, excluding the softmax <Latex>{'$\\sigma$'}</Latex> function, is:</p>
            <Latex delimiters={[{ left: '$$', right: '$$', display: true }]}>{'$$\\mathbf{\\hat{y}} = \\mathbf{L}\\mathbf{x} + \\mathbf{b}$$'}</Latex>
            <p>Where <Latex>{'\\(\\mathbf{L}\\)'}</Latex> corresponds to the weights of the last layer with size <Latex>{'\\(10 \\times 30\\)'}</Latex>, <Latex>{'\\(\\mathbf{x}\\)'}</Latex> is the input from the previous layer with size <Latex>{'\\(30 \\times 1\\)'}</Latex>, and <Latex>{'\\(\\mathbf{b}\\)'}</Latex> is the layer's biases with size <Latex>{'\\(10 \\times 1\\)'}</Latex>. <Latex>{'\\(\\mathbf{\\hat{y}}\\)'}</Latex> is the model's prediction where <Latex>{'\\(\\sigma(\\mathbf{\\hat{y}})_i\\)'}</Latex> can be interpreted as the probability that the <Latex>{'\\(i\\)'}</Latex>th class is the actual class.</p>
            <p>The model's class prediction is then computed as:</p>
            <Latex delimiters={[{ left: '$$', right: '$$', display: true }]}>{'$$\\hat{p} = \\argmax{(\\mathbf{\\hat{y}})}$$'}</Latex>
            <blockquote>
                <strong>Note:</strong> The Softmax function preserves order so it does not affect the output of the argmax function and is not needed in the circuit.
            </blockquote>
            <h3>Preprocessing</h3>
            <p>At the time of circuit development, Noir libraries like <a href="https://github.com/resurgencelabs/signed_int">Signed Int</a> and <a href="https://github.com/resurgencelabs/fraction">Fraction</a> didn't exist. As a result, to circumvent Noir's lack of native signed integers and fraction/floating-points, the input, weights, and biases were scaled and truncated.</p>
            <p>To account for negatives, a very big positive integer <Latex>{'\\(c\\)'}</Latex> is added element-wise to the weights and biases. Due to the previous layer's ReLu activation function, <Latex>{'\\(\\mathbf{x}\\)'}</Latex> is guaranteed to be positive. To account for floating points, a positive scaler <Latex>{'\\(a\\)'}</Latex> scales the weights, biases, and inputs element-wise. Each element is then floored (equivalent to truncation when <Latex>{'\\(\\geq0\\)'}</Latex>). <Latex>{'\\(\\mathbf{J}\\)'}</Latex> is a matrix of ones with size <Latex>{'\\(10 \\times 30\\)'}</Latex>.</p>
            <Latex delimiters={[{ left: '$$', right: '$$', display: true }]}>{`$$
                \\mathbf{z} = \\lfloor{a\\mathbf{x}}\\rfloor \\qquad
                \\mathbf{W} = \\lfloor{a\\mathbf{L}} + c\\mathbf{J}\\rfloor \\qquad
                \\mathbf{v} = \\lfloor{a^2\\mathbf{b}} + c\\vec{1} \\rfloor \\qquad 
            $$`}</Latex>

            <p>Excluding truncation,</p>

            <Latex delimiters={[{ left: '$$', right: '$$', display: true }]}>{`$$
                \\begin{align*}
                \\mathbf{W}\\mathbf{z} + \\mathbf{v} = (a\\mathbf{L}+c\\mathbf{J})
                (a\\mathbf{x}) + (a^2\\mathbf{b} + c\\vec{1}) \\\\
                = a^2\\mathbf{L}\\mathbf{x} + a^2\\mathbf{b} + ca\\mathbf{J_1}\\mathbf{x} + c\\mathbf{J_2} \\\\
                = a^2\\mathbf{\\hat{y}} + ca\\mathbf{J_1}\\mathbf{x} + c\\vec{1} \\\\
                = a^2\\mathbf{\\hat{y}} + ca\\mathbf{J_1}\\mathbf{x} + c\\vec{1} \\\\
                = a^2\\mathbf{\\hat{y}} + c\\begin{bmatrix}
                    a(\\mathbf{1} \\cdot \\mathbf{x})+1 \\\\
                    a(\\mathbf{1} \\cdot \\mathbf{x})+1 \\\\
                    \\vdots \\\\
                    a(\\mathbf{1} \\cdot \\mathbf{x})+1 \\\\
                    a(\\mathbf{1} \\cdot \\mathbf{x})+1
                \\end{bmatrix} \\\\
                \\equiv a^2\\mathbf{\\hat{y}} + \\mathbf{d},\\quad \\mathbf{d} \\in \\mathbb{R}^{10} \\\\ 
                \\therefore \\quad \\argmax{(a^2\\mathbf{\\hat{y}} + \\mathbf{d})} = \\argmax{(\\mathbf{\\hat{y}})} = \\hat{p}
                \\end{align*}
            $$`}</Latex>

            <h3>Commitment</h3>
            <p className={styles.buffer}>After a digit is selected, the user generates a public commitment <Latex>{'$c_x$'}</Latex> equivalent to the pederson hash of the input <Latex>{'$\\mathbf{x}$'}</Latex>. The circuit then checks the constraint <Latex>{'$(\\text{hash}{(\\mathbf{x})} == c_x)$'}</Latex> which ensures that model's prediction corresponds to that commited input.</p>
        </div>
      );
}