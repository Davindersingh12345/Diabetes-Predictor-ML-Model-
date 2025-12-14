document.getElementById("predictionForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let formData = new FormData(this);
    let data = {};

    formData.forEach((value, key) => {
        data[key] = value;
    });

    let response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    let result = await response.json();

    document.getElementById("result").innerText =
        result.prediction === 1
        ? "⚠️ Diabetic Detected"
        : "✅ No Diabetes Detected";
});
