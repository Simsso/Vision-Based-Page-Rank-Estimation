const storage = (() => {
    const CORRECT_CTR_KEY = 'CORRECT_CTR', TOTAL_CTR_KEY = 'TOTAL_CTR';

    function getInt(key) {
        const rawVal = localStorage.getItem(key);
        if (rawVal == null) return 0;
        return parseInt(rawVal, 10);
    }

    function getCorrect() { return getInt(CORRECT_CTR_KEY); }

    function getTotal() { return getInt(TOTAL_CTR_KEY); }

    function addCorrect() {
        let correctCtr = getCorrect();
        correctCtr += 1;
        localStorage.setItem(CORRECT_CTR_KEY, correctCtr);
        incremenTotalCtr();
    }

    function incremenTotalCtr() {
        let totalCtr = getTotal();
        totalCtr += 1;
        localStorage.setItem(TOTAL_CTR_KEY, totalCtr);
    }

    function reset() {
        localStorage.removeItem(CORRECT_CTR_KEY);
        localStorage.removeItem(TOTAL_CTR_KEY);
    }

    function get() {
        return { 
            correctCtr: getCorrect(), 
            totalCtr: getTotal()
        }
    }

    return {
        reset: reset,
        addCorrect: addCorrect,
        addWrong: incremenTotalCtr,
        get: get
    };
})();
