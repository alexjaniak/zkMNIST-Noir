# ZKMNIST-Noir

[![Netlify Status](https://api.netlify.com/api/v1/badges/38671a04-0b2f-417d-a695-8cadf4d59e53/deploy-status)](https://app.netlify.com/sites/zkmnist-noir/deploys)

## Table of Contents
1. [About](#about)
2. [Model & Data](#model--data)
3. [Approach](#approach)
4. [Running Locally](#running-locally)

## About
[Ethereum's](https://ethereum.org/en/) smart contracts enable trustless execution in a decentralized environment. Yet, the limitations in computational power and the transparent aspect of blockchain transactions limit the ability to develop applications that require intensive computation or handle private data, like those used in machine learning. Zero-Knowledge (ZK) proofs aim to free these constraints by allowing off-chain computations to be verified on-chain. Furthermore, they allow the verification of these computations without revealing the underlying data, thereby enabling privacy-preserving smart contracts.

Similarly, these proofs can be used to facilitate the execution of machine learning models in a way that keeps the input data or the model weights confidential. This might be useful when input data is sensitive and private, such as personal biometric information. Likewise, model parameters can be confidential, like those used in biometric authentication systems.

At the same time, it might be essential for downstream entities, such as on-chain smart contracts, that rely on the output of the machine learning model, to be certain that the input was correctly processed by the ML model to yield the claimed output. This ensures both privacy and reliability in the application of these models in sensitive areas.
  
This web app is an example of such zero knowledge machine learning. The app uses a neural network trained on the [MNIST dataset](https://paperswithcode.com/dataset/mnist) (A collection of handwritten digits), that classifies a selected digit and then generates proof of the classification while keeping the inputs (the pixels) private. The proof can then be verified on or off-chain.

The ZK proving system is written using [Aztec Network’s](https://aztec.network/) ZK domain specific language [Noir](https://noir-lang.org/). The web application was built from Aztec's [Noir Starter](https://github.com/noir-lang/noir-starter/tree/main/next-hardhat). It uses [Next.js](https://nextjs.org/) as the frontend framework, 
[Hardhat](https://hardhat.org/) to deploy and test, and [TensorFlow/Keras](https://www.tensorflow.org/) for model training and prediction.

Inspired by [0xPARC's ZK Machine Learning](https://0xparc.org/blog/zk-mnist).

## Model & Data

The MNIST dataset is a collection of 70,000 handwritten digits ranging from 0 to 9, commonly used for training and testing in the field of machine learning. It was created by merging samples from a dataset of American Census Bureau employees and high school students to ensure a diverse set of handwriting styles. Each image is grayscale, 28x28 pixels, and labeled with the corresponding digit it represents. MNIST was designed to be a benchmark dataset to evaluate the performance of algorithms in accurately recognizing and classifying handwritten digits. It has become a standard for evaluating machine learning techniques and therefore a perfect dataset to test Noir's capabilities for zero knowledge machine learning.

![Example of MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/f/f7/MnistExamplesModified.png)

Unlike 0xPARC's CNN approach, the chosen model architecture is a simple dense network:
| Layer (type)   | Output Shape | Param #   |
|----------------|--------------|-----------|
| flatten (Flatten) | (None, 784) | 0         |
| dense (Dense)     | (None, 300) | 235,500   |
| dense_1 (Dense)   | (None, 100) | 30,100    |
| dense_2 (Dense)   | (None, 30)  | 3,030     |
| dense_3 (Dense)   | (None, 10)  | 310       |
| Total params: 268,940 (1.03 MB) | Trainable params: 268,940 (1.03 MB) | Non-trainable params: 0 (0.00 Byte) |

The model uses the ReLu activation function between dense layers and a Softmax function after the last layer. The model was trained using Stochastic Gradient Descent and uses Sparse Categorical Cross Entropy for the loss function. After training, the model achieved 97.52% accuracy on the test set. 

## Approach
The approach used was similar to 0xPARC's implementation with a few alterations to account for changes in the model and Noir's quirks. Like in 0xPARC's demo, and to simplify the circuit, only the last layer of the model was implemented as a zk-SNARK. Noir's simple Rust-like syntax made writing the circuit easy. 

The computation for the neural network's last layer forward pass implemented as a circuit, excluding the Softmax $\sigma$ function, is:
$$\mathbf{\hat{y}} = \mathbf{L}\mathbf{x} + \mathbf{b}$$

Where $\mathbf{L}$ corresponds to the weights of the last layer with size $10 \times 30$, $\mathbf{x}$ is the input from the previous layer with size $30 \times 1$, and $\mathbf{b}$ is the layer's biases with size $10 \times 1$. $\mathbf{\hat{y}}$ is the model's prediction where $\sigma(\mathbf{\hat{y}})_i$ can be interpreted as the probability that the $i$th class is the actual class.

The model's class prediction is then computed as:
$$\hat{p} = \text{argmax}{(\mathbf{\hat{y}})}$$

> **Note:** The Softmax function preserves order so it does not effect the output of the argmax function and is not needed in the circuit.
### Preprocessing

At the time of circuit development, Noir libraries like [Signed Int](https://github.com/resurgencelabs/signed_int) and [Fraction](https://github.com/resurgencelabs/fraction) didn't exist. As a result, to circumvent Noir's lack of native signed integers and fraction/floating-points, the input, weights, and biases were scaled and truncated.  

To account for negatives, a very big positive integer $c$ is added element-wise to the weights and biases. Due the previous layer's ReLu activation function, $\mathbf{x}$ is guaranteed to be positive. To account for floating points, a positive scaler $a$ scales the weights, biases, and inputs element-wise. Each element is then floored (equivalent to truncation when $\geq0$). $\mathbf{J}$ is a matrix of ones with size $10 \times 30$.

$$\begin{align}\mathbf{z} = \lfloor{a\mathbf{x}}\rfloor \qquad\mathbf{W} = \lfloor{a\mathbf{L}} + c\mathbf{J}\rfloor \qquad\mathbf{v} = \lfloor{a^2\mathbf{b}} + c\vec{1}\rfloor \end{align}$$

Excluding truncation,

$$
\begin{align*}
  \mathbf{W}\mathbf{z} + \mathbf{v} = (a\mathbf{L}+c\mathbf{J})(a\mathbf{x}) + (a^2\mathbf{b} + c\vec{1}) \\
  = a^2\mathbf{L}\mathbf{x} + a^2\={b} + ca\mathbf{J}\mathbf{x} + c\mathbf{J_2} \\
  = a^2\mathbf{\hat{y}} + ca\mathbf{J_1}\mathbf{x} + c\vec{1} \\
  = a^2\mathbf{\hat{y}} + ca\mathbf{J_1}\mathbf{x} + c\vec{1} \\\\
  = a^2\mathbf{\hat{y}} + c\begin{bmatrix}    
    a(\mathbf{1} \cdot \mathbf{x})+1 \\    
    a(\mathbf{1} \cdot \mathbf{x})+1 \\    
    \vdots \\    
    a(\mathbf{1} \cdot \mathbf{x})+1 \\    
    a(\mathbf{1} \cdot \mathbf{x})+1
    \end{bmatrix} \\\\
  \equiv a^2\mathbf{\hat{y}} + \mathbf{d},\quad \mathbf{d} \in \mathbb{R}^{10} \\
  \therefore \quad \text{argmax}{(a^2\mathbf{\hat{y}} + \mathbf{d})} = \text{argmax}{(\mathbf{\hat{y}})} = \hat{p}
\end{align*}
$$


### Commitment
After a digit is selected, the user generates a public commitment $c_x$ equivalent to the pederson hash of the input $\mathbf{x}$. The circuit then checks the constraint $(\text{hash}{(\mathbf{x})} == c_x)$ which ensures that model's prediction corresponds to that commited input.

## Running Locally

1. Install [yarn](https://yarnpkg.com/) (tested on yarn v1.22.19)

2. Install [Node.js ≥v18](https://nodejs.org/en) (tested on v18.17.0)

3. Install [noirup](https://noir-lang.org/getting_started/nargo_installation/#option-1-noirup) with

   ```bash
   curl -L https://raw.githubusercontent.com/noir-lang/noirup/main/install | bash
   ```

4. Install Nargo v0.17.0 with

   ```bash
   noirup -v 0.17.0
   ```

If you had a lower version of `nargo` installed previously and are running into errors when
compiling, you may need to uninstall it, instructions
[here](https://noir-lang.org/getting_started/nargo_installation#uninstalling-nargo).

5. Install dependencies with

   ```bash
   yarn
   ```

6. Start a local development EVM at <http://localhost:8545> with:

   ```bash
   npx hardhat node
   ```

   or if foundry is preferred, with

   ```bash
   anvil
   ```

## Deploy locally

1. Start a local development EVM at <http://localhost:8545> with:

   ```bash
   npx hardhat node
   ```

2. Build the project and deploy contracts to the local development chain with:

   ```bash
   NETWORK=localhost yarn build
   ```

   > **Note:** If the deployment fails, try removing `yarn.lock` and reinstalling dependencies with
   > `yarn`.

3. Once your contracts are deployed and the build is finished, start the web app with:

   ```bash
   yarn start
   ```

## Deploy on Sepolia

1. Create a `.env` file. Then create and fill enviromental variables `SEPOLIA_DEPLOYER_PRIVATE_KEY` & `SEPOLIA_ALCHEMY_KEY` with your corresponding Sepolia private key and Alchemy API key.

   > **Note:** You can also add other networks to `hardhat.config.ts` and their corresponding keys in `.env`.

2. Make sure you have funds in your deployer account. Build the project and deploy to Sepolia with:
   ```bash
   NETWORK=sepolia yarn build
   ```
   > **Note:** If you want to use another network, you can deploy there by changing the `NETWORK` environment variable corresponding to the network in `hardhat.config.ts`
