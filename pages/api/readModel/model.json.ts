import fs from 'fs/promises';
import path from 'path';

export default async function handler(req, res) {
  //const { filename } = req.query;
  const filename = "model.json"

  try {
    const projectRoot = process.cwd();
    const filePath = path.join(projectRoot, 'utils/ml/model', filename);
    console.log(filePath)
    const data = await fs.readFile(filePath, 'utf-8');
    res.status(200).send(data);
  } catch (error) {
    res.status(500).send({ error: 'Error reading file' });
  }
}
