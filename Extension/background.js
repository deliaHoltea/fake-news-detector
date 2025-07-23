chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "check-news") {
    fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: request.title,
        text: request.text
      })
    })
    .then(res => res.json())
    .then(data => sendResponse(data))
    .catch(err => sendResponse({ error: err.toString() }));

    // return true to keep sendResponse valid after async
    return true;
  }
});
