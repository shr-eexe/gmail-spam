document.getElementById("classify").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: extractEmailText
  }, async (injectionResults) => {
    if (!injectionResults || !injectionResults[0].result) {
      document.getElementById("result").innerText = "⚠️ No email detected.";
      return;
    }
    let emailText = injectionResults[0].result;

    // Send to FastAPI backend
    let response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: [emailText] })
    });

    let data = await response.json();
    let prediction = data.predictions[0] === 1 ? "Spam" : "Not Spam";
    let prob = (data.probabilities[0] * 100).toFixed(2);

    document.getElementById("result").innerText =
      `Prediction: ${prediction}\nConfidence: ${prob}%`;
  });
});

function extractEmailText() {
  // Gmail’s main email body container
  let body = document.querySelector("div.a3s.aiL");
  if (body) return body.innerText;

  // Fallback for some Gmail layouts
  body = document.querySelector('div[dir="ltr"]');
  return body ? body.innerText : null;
}

