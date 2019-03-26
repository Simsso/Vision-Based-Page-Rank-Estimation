const apiReached = api.status();
ui.setAPIStatus(apiReached);

let tuple = null;
let correctCtr = 0, totalCtr = 0;
let selectionMade = false;

events.on('NEXT_TUPLE_REQ', async () => {
    tuple = await api.nextTuple('v2');
    selectionMade = false;
    ui.showTupleV2(tuple, false);
});

events.on('SELECTION_MADE', async (elemId) => {
    if (selectionMade) {
        return;
    }
    selectionMade = true;
    const correct = tuple[elemId].rank < tuple[1-elemId].rank;
    ui.showTupleV2(tuple, true);

    alert(correct ? "Correct!" : "That was wrong!");

    if (correct) {
        correctCtr += 1;
    }
    totalCtr += 1;

    ui.showScore(correctCtr, totalCtr);
});


// init
events.emit('NEXT_TUPLE_REQ');
