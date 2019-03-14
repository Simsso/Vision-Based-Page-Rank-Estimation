const express = require('express');
const Router = express.Router;
const fs = require('fs');
const path = require('path');
const DatasetV1 = require('../dataset/v1');
const getRankDomainMap = require('../dataset/domains')

async function dataModule() {
    const rankDomainMap = await getRankDomainMap(process.env.DOMAIN_LIST);
    const datav1 = await DatasetV1.fromPath(process.env.DATASET_V1_PATH, rankDomainMap);

    const router = Router();

    router.get('/v1', (req, res) => {
        return res.json(dataV1.listOfFiles);
    });

    router.get('/v1/randtuple', (req, res) => {
        const tuple = datav1.getRandomFileTuple();
        return res.json(tuple);
    });

    router.get('/v1/img/:name', (req, res) => {
        const fileName = req.params.name;
        if (datav1.listOfFiles.indexOf(fileName) === -1) {
            return res.status(404).send();
        }
        
        const filePath = path.join(datav1.rootDir, fileName);
        const type = 'image/jpeg';
        const s = fs.createReadStream(filePath);

        s.on('open', () => {
            res.set('Content-Type', type);
            s.pipe(res);
        });
        s.on('error', () => {
            res.set('Content-Type', 'text/plain');
            return res.status(404).end();
        });
    });

    return router;
}

module.exports = dataModule;