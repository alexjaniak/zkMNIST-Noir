import fs from 'fs';
import csv from 'csv-parser';

interface DataSet { 
    labels: number[];
    data: number[];
    size: number;
}

class GenMNISTSample {
    static async readDataFromCSV(filePath: string): Promise<DataSet> {        
        try {
            const labels: number[] = [];
            const data: number[] = []
            const fileStream = fs.createReadStream(filePath);
        
            fileStream.pipe(csv())
              .on('data', (row) => {
                const arrRow: string[] = Object.keys(row).map(key => row[key]);
                labels.push(+arrRow[0]) // push lable
                // @ts-ignore
                data.push(arrRow.slice(1).map((value) => +value)) // push data
            });
        
            await new Promise((resolve, reject) => {
              fileStream.on('end', resolve);
              fileStream.on('error', reject);
            });
            
            console.log(`CSV file reading completed for file: ${filePath}`);
            return {'labels': labels, 'data': data, 'size': labels.length}

          } catch (error) {
            throw error;
        }
    }

    static groupDataByLabel(MNISTSet: DataSet): Record<number, any> {
        const groupedData: Record<number, any> = {};
            
        // Iterate through the data and data to corresponding label
        for (let i = 0; i < MNISTSet.size; i++) {
            if (groupedData[i] == undefined) {
                groupedData[MNISTSet.labels[i]] = MNISTSet.data[i];
            }
        }
      
        return groupedData;
      }

    static writeJSONSample(sample: any, jsonFilePath: string) {
        // Convert the data to a JSON string
        const jsonData = JSON.stringify(sample); // The second argument is for formatting (2 spaces for indentation)

        // Write the JSON string to the file
        try {
            fs.writeFileSync(jsonFilePath, jsonData, 'utf-8');
            console.log(`Data written to ${jsonFilePath}`);
        } catch (error) {
            console.error(`Error writing to ${jsonFilePath}:`, error);
        }
    }

    static readJSONSample(jsonFilePath: string) {
        try {
            const jsonData = fs.readFileSync(jsonFilePath, 'utf-8');
            const parsedData = JSON.parse(jsonData);
            console.log('Parsed JSON Data:', parsedData);
            return parsedData;
        } catch (error) {
            console.error(`Error reading/parsing ${jsonFilePath}:`, error);
        }
    }
}

async function main() {
    const csvPath = "utils/ml/data/mnist_test.csv";
    const JSONFilePath = "utils/ml/data/sample.json";
    const MNISTData = await GenMNISTSample.readDataFromCSV(csvPath);
    const SampleData = GenMNISTSample.groupDataByLabel(MNISTData)
    GenMNISTSample.writeJSONSample(SampleData, JSONFilePath);
}

main();
    



