const apiReached = api.status();
const slowMode = false;

ui.setAPIStatus(apiReached);

let tuple = null;
let selectionMade = false;

events.on('NEXT_TUPLE_REQ', async () => {
    tuple = await api.nextTuple('v2');
    selectionMade = false;
    ui.showTupleV2(tuple, false);
    ui.scrollToBottom();
});

events.on('SELECTION_MADE', async (elemId) => {
    if (selectionMade) {
        return;
    }
    selectionMade = true;
    const correct = tuple[elemId].rank < tuple[1-elemId].rank;
    if (slowMode) {
        ui.showTupleV2(tuple, true);
    }

    if (correct) {
        storage.addCorrect();
    }
    else {
        storage.addWrong();
    }
    updateScore();

    if (slowMode) {
        alert(correct ? "Correct!" : "That was wrong!");
    }
    else {
        events.emit('NEXT_TUPLE_REQ');
    }
});

events.on('RESET_SCORE', () => {
    storage.reset();
    updateScore();
});

function updateScore() {
    let { correctCtr, totalCtr } = storage.get();

    ui.showScore(correctCtr, totalCtr);
}
updateScore();

// init
events.emit('NEXT_TUPLE_REQ');
