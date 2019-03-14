const fs = require('fs');
const readline = require('readline');

async function getLines(filePath) {
    const fileStream = fs.createReadStream(filePath);
  
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });
    
    const lines = [];

    for await (const line of rl) {
        lines.push(line);
    }

    return lines;
}
  
module.exports = async (rankDomainMapFile) => {
    let lines = await getLines(rankDomainMapFile);

    lines = lines.slice(1);  // remove CSV header
    lines = lines.filter(l => l.length);  // remove empty lines
    lines = lines.map(l => l.split(','));  // split CSV lines
    lines = lines.map(parts => {
        if (parts.length !== 3) {
            throw new Error(`Found an invalid line in the rank domain mapping file: ${l}`);
        }
        return {
            rank: parseInt(parts[0]),
            domain: parts[1]
        };
    });

    const rankDomainMap = {};
    lines.forEach(({rank, domain}) => {
        if (rankDomainMap[rank] != null) {
            throw new Error(`Found two domains for the same rank, namely '${rankDomainMap[rank]}' and '${domain}'`);
        }
        rankDomainMap[rank] = domain;
    })

    return rankDomainMap;
}