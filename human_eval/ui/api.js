const api = (() => {
    const basePath = 'api/v1';

    async function get(endpoint) {
        const url = `${basePath}${endpoint}`;
        return fetch(url).then(res => {
            return res.json();
        });
    }

    async function status() {
        try {
            const res = await get('/status');
            return res.status === 'OK';
        }
        catch {
            return false;
        }
    }

    async function nextTuple() {
        return get('/data/v1/randtuple')
    }

    return {
        status: status,
        nextTuple: nextTuple
    };
})();