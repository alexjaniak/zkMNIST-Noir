import { expect } from 'chai';
import hre from 'hardhat';

import { Noir } from '@noir-lang/noir_js';
import { BarretenbergBackend } from '@noir-lang/backend_barretenberg';

import { compile } from '@noir-lang/noir_wasm';
import path from 'path';
import { ProofData } from '@noir-lang/types';

const getCircuit = async (name: string) => {
  const compiled = await compile(path.resolve('circuits', 'src', `${name}.nr`));
  return compiled;
};

describe('It compiles noir program code, receiving circuit bytes and abi object.', () => {
  let noir: Noir;
  let correctProof: ProofData;

  before(async () => {
    const circuit = await getCircuit('main');
    const verifierContract = await hre.ethers.deployContract('UltraVerifier');

    const verifierAddr = await verifierContract.deployed();
    console.log(`Verifier deployed to ${verifierAddr.address}`);

    const backend = new BarretenbergBackend(circuit);
    noir = new Noir(circuit, backend);
  });

  it('Should generate valid proof for correct input', async () => {
    const input = {
      input : [1840,883,4872,4526,0,0,1582,5512,378,1795,2536,1096,2525,6059,8182,4363,6003,776,108,1426,0,667,2382,1914,576,0,1010,0,0,0],
      weights : [999877,1000897,999320,999249,1000244,1000056,999898,1000101,999518,999776,999586,999690,999098,1000433,1000200,1001075,1000792,999518,999782,1000048,999804,1000072,1000494,999737,999075,1000748,1000146,1000522,999293,999423,999307,1000273,999716,1000981,1000209,999774,999192,999751,1000155,1000067,999984,1000049,1000054,1000261,1000306,999807,999678,999892,1000231,1000076,999810,1000256,1000073,1000000,999939,1000373,999742,999707,1000180,999743,999959,1000773,1000626,999279,999918,999721,999971,999557,1000401,1000068,999349,999398,999959,1000316,999646,1000175,999391,1000327,1000636,999702,1000172,999752,999701,1000159,1000107,1000021,999651,1000049,1000162,999840,999710,999623,999712,999530,1000734,1000216,999534,1000379,999237,999998,1000202,1000534,999835,1000355,999388,1000535,1000027,999319,999808,999363,1000723,1000275,1000147,999342,999710,999644,999780,1000304,1000084,1000134,1000064,999985,1000590,1000494,999361,999277,1000359,999885,999498,1000380,999603,1000166,999523,999672,1000170,999353,999647,1001039,999942,1000449,999454,999727,1000814,1000334,999328,999630,999416,1000707,1000502,1000142,1000170,999415,999644,999727,999829,1000599,999365,999889,999899,1000990,1000650,999879,1000333,1000691,1000126,999526,999404,1000801,999424,999411,1000258,999674,999689,999485,1000216,999732,999809,999689,999965,1000250,1000348,999655,1000293,1000018,1000301,1000422,999608,999715,999706,999753,999903,999905,1000731,999521,1000987,999313,999858,999833,999690,999266,999650,1000355,999667,999741,1000244,999630,1000072,1000250,999835,1000061,999655,1000128,1000360,999871,999800,1000189,999867,999885,999888,999572,1000099,999701,999744,1000048,999510,1000085,999620,999422,1001042,999773,1000334,1000714,999751,1000371,1000458,999759,999703,999624,999815,1000738,1000056,1000294,1000030,999649,1000268,999281,1000220,999836,999974,999124,1000283,1000074,999797,999937,1000352,1000366,999700,999645,1000283,1000244,1000605,999706,999510,1000102,1000551,999634,999831,999787,1000287,1000529,1000358,999644,999633,1000276,999522,999605,1000766,999235,999808,1000574,999767,1000204,1000457,999266,1000459,1000615,1000788,999404,1000436,999555,1000602,999643,999531,999692,1000068,999993,1000305,999834,1000704,1000242],
      biases : [978655,1026255,960272,950844,969485,1118794,1031055,1091307,895820,977509],
      class : 7,   
    };

    // Generate proof
    correctProof = await noir.generateFinalProof(input);
    expect(correctProof.proof instanceof Uint8Array).to.be.true;
  });

  it('Should verify valid proof for correct input', async () => {
    const verification = await noir.verifyFinalProof(correctProof);
    expect(verification).to.be.true;
  });

  it('Should fail to generate valid proof for incorrect input', async () => {
    try {
      const input = {
        input : [1840,883,4872,4526,0,0,1582,5512,378,1795,2536,1096,2525,6059,8182,4363,6003,776,108,1426,0,667,2382,1914,576,0,1010,0,0,0],
        weights : [999877,1000897,999320,999249,1000244,1000056,999898,1000101,999518,999776,999586,999690,999098,1000433,1000200,1001075,1000792,999518,999782,1000048,999804,1000072,1000494,999737,999075,1000748,1000146,1000522,999293,999423,999307,1000273,999716,1000981,1000209,999774,999192,999751,1000155,1000067,999984,1000049,1000054,1000261,1000306,999807,999678,999892,1000231,1000076,999810,1000256,1000073,1000000,999939,1000373,999742,999707,1000180,999743,999959,1000773,1000626,999279,999918,999721,999971,999557,1000401,1000068,999349,999398,999959,1000316,999646,1000175,999391,1000327,1000636,999702,1000172,999752,999701,1000159,1000107,1000021,999651,1000049,1000162,999840,999710,999623,999712,999530,1000734,1000216,999534,1000379,999237,999998,1000202,1000534,999835,1000355,999388,1000535,1000027,999319,999808,999363,1000723,1000275,1000147,999342,999710,999644,999780,1000304,1000084,1000134,1000064,999985,1000590,1000494,999361,999277,1000359,999885,999498,1000380,999603,1000166,999523,999672,1000170,999353,999647,1001039,999942,1000449,999454,999727,1000814,1000334,999328,999630,999416,1000707,1000502,1000142,1000170,999415,999644,999727,999829,1000599,999365,999889,999899,1000990,1000650,999879,1000333,1000691,1000126,999526,999404,1000801,999424,999411,1000258,999674,999689,999485,1000216,999732,999809,999689,999965,1000250,1000348,999655,1000293,1000018,1000301,1000422,999608,999715,999706,999753,999903,999905,1000731,999521,1000987,999313,999858,999833,999690,999266,999650,1000355,999667,999741,1000244,999630,1000072,1000250,999835,1000061,999655,1000128,1000360,999871,999800,1000189,999867,999885,999888,999572,1000099,999701,999744,1000048,999510,1000085,999620,999422,1001042,999773,1000334,1000714,999751,1000371,1000458,999759,999703,999624,999815,1000738,1000056,1000294,1000030,999649,1000268,999281,1000220,999836,999974,999124,1000283,1000074,999797,999937,1000352,1000366,999700,999645,1000283,1000244,1000605,999706,999510,1000102,1000551,999634,999831,999787,1000287,1000529,1000358,999644,999633,1000276,999522,999605,1000766,999235,999808,1000574,999767,1000204,1000457,999266,1000459,1000615,1000788,999404,1000436,999555,1000602,999643,999531,999692,1000068,999993,1000305,999834,1000704,1000242],
        biases : [978655,1026255,960272,950844,969485,1118794,1031055,1091307,895820,977509],
        class : 5,   
      };
      const incorrectProof = await noir.generateFinalProof(input);
    } catch (err) {
      // TODO(Ze): Not sure how detailed we want this test to be
      expect(err instanceof Error).to.be.true;
      const error = err as Error;
      expect(error.message).to.contain('Cannot satisfy constraint');
    }
  });
});