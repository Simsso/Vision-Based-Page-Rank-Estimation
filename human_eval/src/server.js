require('dotenv').config();

async function main() {
    const express = require('express');
    const app = express();
    const bodyParser = require('body-parser');
    const morgan = require('morgan');

    // logging
    app.use(morgan('combined'));

    // parse application/x-www-form-urlencoded
    app.use(bodyParser.urlencoded({ extended: false }));

    // parse application/json
    app.use(bodyParser.json());

    app.get('/', (req, res) => {
        res.json({ status: 'OK' });
    });

    const dataRouter = await require('./api/data')();
    app.use('/data', dataRouter);

    const port = process.env.PORT
    app.listen(port, () => console.log('Service running on port ' + port))
}

(async () => {
    try {
        await main();
    } catch (e) {
        console.log(e);
        throw e
    }
})();
