const fs = require('fs');


class DatasetV1 {
    constructor(rootDir, listOfFiles, rankDomainMap) {
        this.rootDir = rootDir;
        this.listOfFiles = listOfFiles;
        this.rankDomainMap = rankDomainMap;
    }

    getSample(idx) {
        const fileName = this.listOfFiles[idx];
        const rank = parseInt(fileName.split('.')[0], 10);
        return {
            file: fileName,
            rank: rank,
            domain: this.rankDomainMap[rank]
        };
    }

    getRandomFile() {
        const arr = this.listOfFiles;
        const idx = Math.floor(Math.random() * arr.length);
        return this.getSample(idx);
    }

    getRandomFileTuple() {
        const arr = this.listOfFiles;
        const idx1 = Math.floor(Math.random() * arr.length);
        let idx2 = Math.floor(Math.random() * (arr.length - 1));
        if (idx2 >= idx1) {
            idx2 += 1;
        }
        return [this.getSample(idx1), this.getSample(idx2)];
    }
}

DatasetV1.fromPath = async function(path, rankDomainMap) {
    return new Promise((resolve, reject) => {
        fs.readdir(path, (err, listOfFiles) => {
            if (err) return reject(err);

            listOfFiles = listOfFiles.filter(s => s.endsWith('.jpg') || s.endsWith('.jpeg'));
            dataset = new DatasetV1(path, listOfFiles, rankDomainMap);

            return resolve(dataset);
        });
    });
};

module.exports = DatasetV1;
