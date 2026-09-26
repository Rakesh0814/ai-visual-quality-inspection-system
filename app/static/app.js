// ==========================================================
// InspectAI Frontend
// Existing FastAPI backend:
//
// GET  /health
// POST /api/inspect
// GET  /api/history
// ==========================================================


const fileInput =
    document.getElementById("fileInput");

const dropZone =
    document.getElementById("dropZone");

const uploadPlaceholder =
    document.getElementById("uploadPlaceholder");

const preview =
    document.getElementById("preview");

const runButton =
    document.getElementById("runButton");

const resultImage =
    document.getElementById("resultImage");

const resultEmpty =
    document.getElementById("resultEmpty");

const resultStatus =
    document.getElementById("resultStatus");

const prediction =
    document.getElementById("prediction");

const confidence =
    document.getElementById("confidence");

const defectProbability =
    document.getElementById("defectProbability");

const backend =
    document.getElementById("backend");

const message =
    document.getElementById("message");

const modelMeta =
    document.getElementById("modelMeta");

const historyBody =
    document.getElementById("historyBody");

const refreshButton =
    document.getElementById("refreshButton");


let selectedFile = null;

let previewUrl = null;



// ==========================================================
// FILE PREVIEW
// ==========================================================

function displaySelectedFile(file) {

    if (!file) {
        return;
    }


    if (!file.type.startsWith("image/")) {

        alert(
            "Please upload an image file."
        );

        return;
    }


    selectedFile = file;


    if (previewUrl) {

        URL.revokeObjectURL(
            previewUrl
        );

    }


    previewUrl =
        URL.createObjectURL(file);


    preview.src =
        previewUrl;


    resultImage.src =
        previewUrl;


    preview.classList.remove(
        "hidden"
    );


    resultImage.classList.remove(
        "hidden"
    );


    uploadPlaceholder.classList.add(
        "hidden"
    );


    resultEmpty.classList.add(
        "hidden"
    );


    resetPredictionDisplay();

}



fileInput.addEventListener(
    "change",

    () => {

        const file =
            fileInput.files[0];

        displaySelectedFile(
            file
        );

    }
);



// ==========================================================
// DRAG DROP
// ==========================================================

dropZone.addEventListener(
    "dragover",

    (event) => {

        event.preventDefault();

        dropZone.style.borderColor =
            "#46b2ff";

    }
);


dropZone.addEventListener(
    "dragleave",

    () => {

        dropZone.style.borderColor =
            "";

    }
);


dropZone.addEventListener(
    "drop",

    (event) => {

        event.preventDefault();

        dropZone.style.borderColor =
            "";

        const file =
            event.dataTransfer.files[0];

        displaySelectedFile(
            file
        );

    }
);



// ==========================================================
// RESET RESULT
// ==========================================================

function resetPredictionDisplay() {

    resultStatus.className =
        "prediction-status waiting";


    prediction.textContent =
        "READY";


    confidence.textContent =
        "—";


    defectProbability.textContent =
        "—";


    backend.textContent =
        "—";


    message.textContent =
        "Image loaded. Run inspection to generate a prediction.";


    modelMeta.textContent =
        "";

}



// ==========================================================
// INSPECTION
// ==========================================================

runButton.addEventListener(
    "click",

    async () => {

        if (!selectedFile) {

            fileInput.click();

            return;

        }


        runButton.disabled =
            true;


        runButton.innerHTML =
            "Inspecting...";


        message.textContent =
            "Running PyTorch model inference...";


        try {

            const formData =
                new FormData();


            formData.append(
                "file",
                selectedFile
            );


            const response =
                await fetch(
                    "/api/inspect",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Inspection failed."
                );

            }


            updateResult(
                data
            );


            await loadHistory();

        }

        catch (error) {

            prediction.textContent =
                "ERROR";


            resultStatus.className =
                "prediction-status defect";


            message.textContent =
                error.message;

        }

        finally {

            runButton.disabled =
                false;


            runButton.innerHTML =
                "Run Inspection <span>→</span>";

        }

    }
);



// ==========================================================
// RESULT UPDATE
// ==========================================================

function updateResult(data) {

    const label =
        String(
            data.label || ""
        ).toUpperCase();


    const confidencePercent =
        Math.round(
            (data.confidence || 0) *
            100
        );


    const defectPercent =
        Math.round(
            (data.defect_probability || 0) *
            100
        );


    prediction.textContent =
        label;


    confidence.textContent =
        `${confidencePercent}%`;


    defectProbability.textContent =
        `${defectPercent}%`;


    backend.textContent =
        data.backend || "pytorch";


    if (label === "PASS") {

        resultStatus.className =
            "prediction-status good";


        resultStatus.querySelector(
            ".prediction-icon"
        ).textContent =
            "✓";


        message.textContent =
            "Bottle classified as GOOD. No learned defect pattern exceeded the decision threshold.";

    }

    else {

        resultStatus.className =
            "prediction-status defect";


        resultStatus.querySelector(
            ".prediction-icon"
        ).textContent =
            "!";


        message.textContent =
            "Bottle classified as DEFECT by the trained image classifier.";

    }


    let meta =
        [];


    if (data.model) {

        meta.push(
            data.model
        );

    }


    if (data.dataset) {

        meta.push(
            data.dataset
        );

    }


    if (
        data.validation_accuracy !==
        null &&
        data.validation_accuracy !==
        undefined
    ) {

        meta.push(
            `Validation accuracy: ${
                Math.round(
                    data.validation_accuracy *
                    100
                )
            }%`
        );

    }


    modelMeta.textContent =
        meta.join("  •  ");

}



// ==========================================================
// HISTORY
// ==========================================================

async function loadHistory() {

    try {

        const response =
            await fetch(
                "/api/history?limit=20"
            );


        const items =
            await response.json();


        historyBody.innerHTML =
            "";


        if (
            !Array.isArray(items) ||
            items.length === 0
        ) {

            historyBody.innerHTML =
                `
                    <tr>

                        <td colspan="5">
                            No inspections yet.
                        </td>

                    </tr>
                `;

            return;

        }


        items.forEach(
            (item) => {

                const row =
                    document.createElement(
                        "tr"
                    );


                const label =
                    String(
                        item.label || ""
                    ).toUpperCase();


                const resultClass =
                    label === "PASS"
                        ? "history-good"
                        : "history-defect";


                const confidenceValue =
                    Math.round(
                        (item.confidence || 0)
                        * 100
                    );


                const date =
                    item.created_at
                        ? new Date(
                            item.created_at
                        ).toLocaleString()
                        : "—";


                row.innerHTML =
                    `

                        <td>
                            ${
                                escapeHtml(
                                    item.filename ||
                                    "image"
                                )
                            }
                        </td>


                        <td
                            class="${resultClass}"
                        >

                            ${
                                label === "PASS"
                                    ? "GOOD"
                                    : "DEFECT"
                            }

                        </td>


                        <td>

                            ${confidenceValue}%

                        </td>


                        <td>

                            ${
                                escapeHtml(
                                    item.backend ||
                                    "pytorch"
                                )
                            }

                        </td>


                        <td>

                            ${date}

                        </td>

                    `;


                historyBody.appendChild(
                    row
                );

            }
        );

    }

    catch (error) {

        console.error(
            "Could not load history:",
            error
        );

    }

}



refreshButton.addEventListener(
    "click",
    loadHistory
);



// ==========================================================
// SECURITY HELPER
// ==========================================================

function escapeHtml(value) {

    const element =
        document.createElement(
            "div"
        );


    element.textContent =
        value;


    return element.innerHTML;

}



// ==========================================================
// HEALTH CHECK
// ==========================================================

async function checkHealth() {

    try {

        const response =
            await fetch(
                "/health"
            );


        const data =
            await response.json();


        if (
            data.model_available === false
        ) {

            message.textContent =
                "No trained model found. Train the model before running inspections.";

        }

    }

    catch (error) {

        console.error(
            "Health check failed:",
            error
        );

    }

}



// ==========================================================
// INITIALIZE
// ==========================================================

checkHealth();

loadHistory();