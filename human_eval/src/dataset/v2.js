const fs = require('fs');
const path = require('path');

const isDirectory = source => fs.lstatSync(source).isDirectory()


class DatasetV2 {
    constructor(rootDir, listOfFolders, rankDomainMap) {
        this.rootDir = rootDir;
        this.listOfFolders = listOfFolders;
        this.rankDomainMap = rankDomainMap;
    }

    async getSample(idx) {
        return new Promise((resolve, reject) => {
            const folderName = this.listOfFolders[idx];
            const rank = parseInt(folderName, 10);
            fs.readdir(path.join(this.rootDir, folderName, 'image'), (err, files) => {
                files = files.filter(s => s.endsWith('.jpg') || s.endsWith('.jpeg'));
                return resolve({
                    folder: folderName,
                    rank: rank,
                    images: files,
                    domain: this.rankDomainMap[rank]
                });

            });
        });
    }

    async getRandomFile() {
        const arr = this.listOfFolders;
        const idx = Math.floor(Math.random() * arr.length);
        return await this.getSample(idx);
    }

    async getRandomTuple() {
        const arr = this.listOfFolders;
        const idx1 = Math.floor(Math.random() * arr.length);
        let idx2 = Math.floor(Math.random() * (arr.length - 1));
        if (idx2 >= idx1) {
            idx2 += 1;
        }
        return [await this.getSample(idx1), await this.getSample(idx2)];
    }
}

DatasetV2.fromPath = async function(rootPath, rankDomainMap) {
    return new Promise((resolve, reject) => {
        fs.readdir(rootPath, (err, listOfFolders) => {
            if (err) return reject(err);

            listOfFolders = listOfFolders.filter(f => isDirectory(path.join(rootPath, f)));
            dataset = new DatasetV2(rootPath, listOfFolders, rankDomainMap);

            return resolve(dataset);
        });
    });
};

module.exports = DatasetV2;
