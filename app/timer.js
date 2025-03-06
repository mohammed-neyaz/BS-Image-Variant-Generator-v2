export function setupTimer() {
    let timerInterval;
    let seconds = 0;
    //const timer = document.getElementById('timer');
    const timerDisplay = document.getElementById('timerDisplay');


    document.addEventListener('generatePrompt', () => {
        // Reset and start timer
        seconds = 0;
        //timer.classList.remove('hidden');
        updateTimerDisplay();
        startTimer();
    });

    document.addEventListener('generateVariants', () => {
        // Reset and start timer
        seconds = 0;
        //timer.classList.remove('hidden');
        updateTimerDisplay();
        startTimer();
    });

    document.addEventListener('variantsGenerated', () => {
        // Stop timer when variants are generated
        stopTimer();
    });

    function startTimer() {
        stopTimer(); // Clear any existing interval
        timerInterval = setInterval(() => {
            seconds++;
            updateTimerDisplay();
        }, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
        }
    }

    function updateTimerDisplay() {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        timerDisplay.textContent = `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
    }
}