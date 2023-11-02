import * as fs from "fs";
import { parse } from "csv-parse";

// MNIST data configuration
export interface DataConfig {
    MNIST_TRAIN_PATH?: String,
    MNIST_TEST_PATH?: String,
    TRAIN_SET_SIZE: number,
    TEST_SET_SIZE: number,
    IMAGE_SIZE: number,
    LABEL_SIZE: number,
}

export const standardConfig: DataConfig = {
    TRAIN_SET_SIZE: 60000,
    TEST_SET_SIZE: 10000,
    IMAGE_SIZE: 784,
    LABEL_SIZE: 1,
}

export interface DataSet { 
    labels: Uint8Array;
    data: Uint8Array;
    size: number;
}

// Populates array with fill from start
function populateArray<T>(arr: any, fill: any, start: number) {
    // start: inclusive
    fill.forEach((element: any) => {
        arr[start] = element;
        start++;
    });
   
}  

export class MNIST {
    dataConfig: DataConfig
    testData!: DataSet;
    trainData!: DataSet;
    trainBatch = 0;
    testBatch = 0;

    constructor(config: DataConfig) {
        this.dataConfig = config;
    }

    // Loads all test data from MNIST_TEST_PATH
    public async loadTestData() {
        this.testData = { 
            labels: new Uint8Array(this.dataConfig.LABEL_SIZE * this.dataConfig.TEST_SET_SIZE),
            data: new Uint8Array(this.dataConfig.IMAGE_SIZE * this.dataConfig.TEST_SET_SIZE),
            size: this.dataConfig.TEST_SET_SIZE
        };
        
        // @ts-ignore
        await this._readData(this.dataConfig.MNIST_TEST_PATH, this.testData);
    }

    // Loads all train data from MNIST_TRAIN_PATH
    public async loadTrainData() {
        this.trainData = { 
            labels: new Uint8Array(this.dataConfig.LABEL_SIZE * this.dataConfig.TRAIN_SET_SIZE),
            data: new Uint8Array(this.dataConfig.IMAGE_SIZE * this.dataConfig.TRAIN_SET_SIZE),
            size: this.dataConfig.TEST_SET_SIZE
        };

        // @ts-ignore
        await this._readData(this.dataConfig.MNIST_TRAIN_PATH, this.trainData);
    }

    // Reads data from PATH and into ret
    private async _readData(PATH: string, ret: DataSet): Promise<void> {
        const config = this.dataConfig
        return new Promise((resolve, reject) => {
            let labelCount = 0;
            let dataCount = 0;
    
            fs.createReadStream(PATH)
                .pipe(parse({ delimiter: ",", from_line: 2 }))
                .on("data", row => {
                    ret.labels[labelCount] = row[0];
                    populateArray(ret.data, row.slice(1, dataCount+this.dataConfig.IMAGE_SIZE), dataCount)
                    labelCount += this.dataConfig.LABEL_SIZE;
                    dataCount += this.dataConfig.IMAGE_SIZE;
                })
                .on("end", () => {
                    resolve();
                })
                .on("error", (err: Error) => {
                    console.error(`ERROR: Failed to read data from {PATH}.`);
                    reject(err);
                });
        });
        
    }

    // Returns next n testing values
    public getNextTestBatch(n: number = 1): any {
        if (this.testData == null) {
            console.error("ERROR: No training data available. Try calling loadTestData().");
            return -1;
        }
        if (this.testBatch + n >= this.dataConfig.TEST_SET_SIZE) {
            console.error("ERROR: No more testing data available.");
            return -1;
        }   
        let ret = {
            labels: this.testData.labels.slice(this.testBatch, this.testBatch + n*this.dataConfig.LABEL_SIZE),
            data: this.testData.data.slice(this.testBatch, this.testBatch + n*this.dataConfig.IMAGE_SIZE),
            size: n
        };
        this.testBatch += n
        return ret;
    }

    // Returns next n testing values
    public getNextTrainBatch(n: number = 1): any {
        if (this.trainData == null) {
            console.error("ERROR: No training data available. Try calling loadTestData().");
            return -1;
        }
        if (this.trainBatch + n >= this.dataConfig.TRAIN_SET_SIZE) {
            console.error("ERROR: Training data limit exceeded.");
            return -1;
        }   

        let ret = {
            labels: this.trainData.labels.slice(this.trainBatch, this.trainBatch + n*this.dataConfig.LABEL_SIZE),
            data: this.trainData.data.slice(this.trainBatch, this.trainBatch + n*this.dataConfig.IMAGE_SIZE),
            size: n
        };

        this.trainBatch += n;
        return ret;
    }
}



