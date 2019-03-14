const ui = (() => {

    // register events
    document.getElementById('nexttuple').addEventListener('click', () => events.emit('NEXT_TUPLE_REQ'));
    [...document.getElementsByClassName('pagepreview')].forEach(prevImg => {
        prevImg.addEventListener('click', event => {
            const elem = event.toElement;
            const elemId = parseInt(elem.attributes['data-id'].value, 10);
            events.emit('SELECTION_MADE', elemId);
        });
    })

    function showImg(id, imgName) {
        document.getElementById(`page${id}img`).src = `api/v1/data/v1/img/${imgName}`;
    }

    function showURL(id, domain) {
        const aTag = document.getElementById(`page${id}url`);
        aTag.href = `http://${domain}`
        aTag.innerHTML = domain;
    }

    function setRankText(id, rankText) {
        document.getElementById(`page${id}rank`).innerHTML = rankText;
    }

    function showSample(id, {file, rank, domain}, showRank) {
        showImg(id, file);
        showURL(id, domain);
        setRankText(id, showRank ? rank : '');
    }

    function showTuple(tuple, showRank) {
        showSample(0, tuple[0], showRank)
        showSample(1, tuple[1], showRank)
    }

    function showScore(correctCtr, totalCtr) {
        const acc = (correctCtr / totalCtr * 100).toFixed(2);
        document.getElementById('score').innerHTML = `Your accuracy is ${acc}%`;
    }

    return {
        setAPIStatus: status => document.getElementById('statusindicator').innerHTML = status ? "API reached" : "API error",
        showTuple: showTuple,
        showScore: showScore
    };
})();
