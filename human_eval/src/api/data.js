const express = require('express');
const Router = express.Router;
const fs = require('fs');
const path = require('path');
const DatasetV1 = require('../dataset/v1');
const DatasetV2 = require('../dataset/v2');
const getRankDomainMap = require('../dataset/domains');

async function dataModule() {
    const rankDomainMap = await getRankDomainMap(process.env.DOMAIN_LIST);
    const datav1 = await DatasetV1.fromPath(process.env.DATASET_V1_PATH, rankDomainMap);
    const datav2 = await DatasetV2.fromPath(process.env.DATASET_V2_PATH, rankDomainMap);

    const router = Router();

    router.get('/v1', (req, res) => res.json(datav1.listOfFiles));
    router.get('/v2', (req, res) => res.json(datav2.listOfFolders));

    router.get('/v1/randtuple', (req, res) => res.json(datav1.getRandomFileTuple()));
    router.get('/v2/randtuple', async (req, res) => res.json(await datav2.getRandomTuple()));

    function streamImg(imgPath, res) {
        const type = 'image/jpeg';
        const s = fs.createReadStream(imgPath);

        s.on('open', () => {
            res.set('Content-Type', type);
            s.pipe(res);
        });
        s.on('error', () => {
            res.set('Content-Type', 'text/plain');
            return res.status(404).end();
        });
    }

    router.get('/v1/img/:name', (req, res) => {
        const fileName = req.params.name;
        if (datav1.listOfFiles.indexOf(fileName) === -1) {
            return res.status(404).send();
        }
        
        const filePath = path.join(datav1.rootDir, fileName);

        return streamImg(filePath, res);
    });

    router.get('/v2/img/:rank/:name', (req, res) => {
        const folderName = req.params.rank;
        const fileName = req.params.name;
        if (datav2.listOfFolders.indexOf(folderName) === -1) {
            return res.status(404).send();
        }

        const filePath = path.join(datav2.rootDir, folderName, 'image', fileName);

        return streamImg(filePath, res);
    });

    return router;
}

module.exports = dataModule;