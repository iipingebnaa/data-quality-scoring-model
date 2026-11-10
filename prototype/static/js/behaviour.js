
document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("onboardingForm");

    if (!form) return;

    const pageStartTime = performance.now();

    const metrics = {
        totalKeyPresses: 0,
        correctionCount: 0,
        flightTimes: [],
        dwellTimes: [],
        inputLatencies: []
    };

    let firstKeyPressTime = null;
    let lastKeyRelease = null;
    let lastInputTime = null;

    const heldKeys = new Map();

    const ignoredKeys = [
        "Shift", "Control", "Alt", "Tab",
        "Enter", "Escape", "ArrowLeft",
        "ArrowRight", "ArrowUp", "ArrowDown"
    ];

    // Measure keyboard activity without recording typed content.
    form.addEventListener("keydown", function (event) {

        if (!event.target.matches("input, textarea")) return;
        if (ignoredKeys.includes(event.key)) return;

        const now = performance.now();

        if (!heldKeys.has(event.code)) {

            metrics.totalKeyPresses++;

            if (firstKeyPressTime === null) {
                firstKeyPressTime = now;
            }

            if (lastKeyRelease !== null) {
                metrics.flightTimes.push(now - lastKeyRelease);
            }

            heldKeys.set(event.code, now);
        }
    });

    // Measure how long each key is held.
    form.addEventListener("keyup", function (event) {

        if (ignoredKeys.includes(event.key)) return;

        const now = performance.now();
        const start = heldKeys.get(event.code);

        if (start !== undefined) {
            metrics.dwellTimes.push(now - start);
            heldKeys.delete(event.code);
        }

        lastKeyRelease = now;
    });

    // Count deletion actions and measure intervals between input events.
    form.addEventListener("input", function (event) {

        const now = performance.now();

        if (
            event.inputType === "deleteContentBackward" ||
            event.inputType === "deleteContentForward"
        ) {
            metrics.correctionCount++;
        }

        if (lastInputTime !== null) {
            metrics.inputLatencies.push(now - lastInputTime);
        }

        lastInputTime = now;
    });

    function average(values) {

        if (values.length === 0) return 0;

        return values.reduce((sum, value) => sum + value, 0)
            / values.length;
    }

    function addHiddenField(name, value) {

        let field = form.querySelector(
            `input[name="${name}"]`
        );

        if (!field) {
            field = document.createElement("input");
            field.type = "hidden";
            field.name = name;
            form.appendChild(field);
        }

        field.value = String(value);
    }

    // Attach the behavioural measurements before submission.
    form.addEventListener("submit", function () {

        const now = performance.now();

        const typingDuration = firstKeyPressTime === null
            ? 0
            : (now - firstKeyPressTime) / 60000;

        const summary = {

            avg_typing_cadence:
                typingDuration > 0
                ? metrics.totalKeyPresses / typingDuration
                : 0,

            avg_flight_time_ms:
                average(metrics.flightTimes),

            avg_input_latency_ms:
                average(metrics.inputLatencies),

            avg_dwell_time_ms:
                average(metrics.dwellTimes),

            correction_count:
                metrics.correctionCount,

            session_duration_seconds:
                (now - pageStartTime) / 1000
        };

        Object.entries(summary).forEach(function ([name, value]) {
            addHiddenField(name, value);
        });

        // Retain only the numerical summary for this browser session.
        sessionStorage.setItem(
            "behaviour_metrics",
            JSON.stringify(summary)
        );

        console.log("Behavioural measurements:", summary);
    });

});
