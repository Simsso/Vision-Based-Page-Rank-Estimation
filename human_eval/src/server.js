require('dotenv').config();
const express = require('express');
const Router = express.Router;
const bodyParser = require('body-parser');
const morgan = require('morgan');
const path = require('path');

async function main() {
    const app = express();
    const api = Router();

    // logging
    app.use(morgan(':method :url :status :res[content-length] - :response-time ms'));

    // parse application/x-www-form-urlencoded
    app.use(bodyParser.urlencoded({ extended: false }));

    // parse application/json
    app.use(bodyParser.json());

    api.get('/status', (req, res) => res.json({ status: 'OK' }));

    const dataRouter = await require('./api/data')();
    api.use('/data', dataRouter);

    app.use('/api/v1', api);

    app.use('/', express.static('ui'));  // serve UI

    const port = process.env.PORT
    app.listen(port, '0.0.0.0', () => console.log('Service running on port ' + port))
}

(async () => {
    try {
        await main();
    } catch (e) {
        console.log(e);
        throw e
    }
})();
