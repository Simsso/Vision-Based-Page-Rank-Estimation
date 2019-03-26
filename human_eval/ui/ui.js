const ui = (() => {

    // register events
    document.onkeypress = e => {  // hotkeys
        e = e || window.event;
        if (e.keyCode === 110) {  // 'n' key
            events.emit('NEXT_TUPLE_REQ')
        }
        else if (e.keyCode === 49) {  // '1' key
            events.emit('SELECTION_MADE', 0);
        }
        else if (e.keyCode === 50) {  // '2' key
            events.emit('SELECTION_MADE', 1);
        }
    };

    // next tuple request
    document.getElementById('nexttuple').addEventListener('click', () => events.emit('NEXT_TUPLE_REQ'));

    // selection made event
    [...document.getElementsByClassName('pagepreview')].forEach(prevImg => {
        prevImg.addEventListener('click', event => {
            let elem = event.toElement;
            if (elem.attributes['data-id'] == null) {
                elem = elem.parentElement;
            }
            const elemId = parseInt(elem.attributes['data-id'].value, 10);
            events.emit('SELECTION_MADE', elemId);
        });
    });

    // reset score button
    document.getElementById('score-reset-link').addEventListener('click', () => events.emit('RESET_SCORE'));

    function showImg(id, imgName) {
        document.getElementById(`page${id}img`).src = `api/v1/data/v1/img/${imgName}`;
    }

    function showImages(id, folder, images) {
        const wrapperDiv = document.getElementById(`page${id}screenshots`);
        wrapperDiv.innerHTML = '';  // clear children
        for (let i = 0; i < images.length; i++) {
            const img = document.createElement('img');
            img.setAttribute('src', `api/v1/data/v2/img/${folder}/${images[i]}`);
            img.setAttribute('alt', 'screenshot');
            wrapperDiv.appendChild(img);
        }
    }

    function showURL(id, domain) {
        const aTag = document.getElementById(`page${id}url`);
        aTag.href = `http://${domain}`
        aTag.innerHTML = domain;
    }

    function setRankText(id, rankText) {
        document.getElementById(`page${id}rank`).innerHTML = rankText;
    }

    function showSampleV1(id, {file, rank, domain}, showRank) {
        showImg(id, file);
        showURL(id, domain);
        setRankText(id, showRank ? rank : '');
    }

    function showTupleV1(tuple, showRank) {
        showSampleV1(0, tuple[0], showRank)
        showSampleV1(1, tuple[1], showRank)
    }

    function showSampleV2(id, {domain, folder, images, rank}, showRank) {
        showImages(id, folder, images);
        showURL(id, domain);
        setRankText(id, showRank ? rank : '');
    }

    function showTupleV2(tuple, showRank) {
        showSampleV2(0, tuple[0], showRank)
        showSampleV2(1, tuple[1], showRank)
    }

    function showScore(correctCtr, totalCtr) {
        let acc = 'unknown';
        if (totalCtr != 0) {
            acc = (correctCtr / totalCtr * 100).toFixed(2);
        }
        document.getElementById('score').innerHTML = `Your accuracy is ${acc}%`;
    }

    return {
        setAPIStatus: status => document.getElementById('statusindicator').innerHTML = status ? "API reached" : "API error",
        showTupleV1: showTupleV1,
        showTupleV2: showTupleV2,
        showScore: showScore
    };
})();
