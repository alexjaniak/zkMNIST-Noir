import * as tf from "@tensorflow/tfjs";

export async function fetchJSONData(url: string) {
    console.log(__dirname);
    try {
      const response = await fetch(url);
  
      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.status}`);
      }
  
      const data = await response.json();
      return data
  
      // You can now work with the 'data' object, which contains the parsed JSON
    } catch (error) {
      console.error('Error:', error);
    }
  }

function nLayerModel(model: tf.LayersModel, nLayers: number) {
  // Get the layers from the input layer up to and including the nth layer
  const layersToExtract = model.layers.slice(0, nLayers);

  // Define a new model with the selected layers
  const nLayerModel = tf.model({
    inputs: model.inputs,
    outputs: layersToExtract[layersToExtract.length - 1].output,
  });

  return nLayerModel
}

function scaleInput(input: tf.Tensor | tf.Tensor[], weight: number, bias: number, ) {
  // Scale input using a weight & bias
  // @ts-ignore
  return input.mul(weight).add(bias).toInt();
}

function computeOutputLayer(input: tf.Tensor, weights: tf.Tensor, biases: tf.Tensor) {
  // Compute outuput layer manually (without softmax)
  return tf.matMul(input, weights).add(biases).argMax(1);
}

export function forwardPass(model: tf.LayersModel, input: tf.Tensor) {
    // Compute forward pass through model, computing the last layer manually

    /* DEBUG & PRINT
    // get inference from full model
    const fullModelInference = fullModel.predict(input).argMax(1);

    // @ts-ignore
    fullModelInference.data().then(arr => console.log("Full Model Inference: ", arr));

    // @ts-ignore
    sawedModelOutput.data().then(arr => console.log("Sawed Model Inference: ", arr));
    */ 

    const sawedModel = nLayerModel(model, model.layers.length-1);
    const sawedModelOutput = sawedModel.predict(input);
    // @ts-ignore
    const output = model.predict(input).argMax(1);

    // Compute last layer manually
    //const BIG_INT = 100000000;
    const BIG_INT = 1000000;
    const [weights, biases] = model.layers[model.layers.length-1].getWeights();

    const scaledInput = scaleInput(sawedModelOutput, 1000, 0)
    const scaledWeights = scaleInput(weights, 1000, BIG_INT);
    const scaledBias = scaleInput(biases, 1000*1000, BIG_INT);
    //const output = computeOutputLayer(scaledInput, scaledWeights, scaledBias);

    return [scaledInput, scaledWeights, scaledBias, output];
}