document.addEventListener("DOMContentLoaded", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const title = document.querySelector("title")?.innerText || document.title || "";
      const paragraphs = Array.from(document.querySelectorAll("p")).map(p => p.innerText).join(" ");
      return { title, text: paragraphs };
    }
  }, async (results) => {
    const { title, text } = results[0].result;
    document.getElementById("title").innerText = title;

    // === CHECK NEWS ===
    document.getElementById("checkBtn").addEventListener("click", () => {
      document.getElementById("result").innerText = "Checking...";
      chrome.runtime.sendMessage({ type: "check-news", title, text }, (response) => {
        const resultBox = document.getElementById("result");

        if (response.error) {
          resultBox.innerText = "Error: " + response.error;
          resultBox.style.color = "black";
        } else {
          const isFake = response.label.toUpperCase() === "FAKE";
          const color = isFake ? "red" : "green";
          const labelText = isFake ? "This news is probably FAKE." : "This news is probably REAL.";

          const realPercent = (response.score_real * 100).toFixed(1);
          const fakePercent = (response.score_fake * 100).toFixed(1);

          resultBox.style.color = color;
          resultBox.innerHTML = `
            <strong style="color:${color}">${labelText}</strong><br>
            <small>Real: ${realPercent}% | Fake: ${fakePercent}%</small>
          `;

          // ✅ Afișăm zona de hint doar după verificare AI
          document.getElementById("hintSection").style.display = "block";
        }
      });
    });

    // === GENERATE HINT ===
    fetch("http://localhost:5000/generate-hint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text })
    })
      .then(res => res.json())
      .then(data => {
        document.getElementById("factHint").value = data.hint;
      })
      .catch(err => {
        document.getElementById("factHint").placeholder = "Failed to generate hint.";
      });

    // === FACT CHECK API SEARCH ===
    document.getElementById("searchFact").addEventListener("click", async () => {
      const query = document.getElementById("factHint").value.trim();
      const resultsBox = document.getElementById("factCheckResults");
      resultsBox.innerHTML = "Searching...";

      const apiKey = "AIzaSyBh6Vo883ZTuwCrAQW6VwA1ckYWyyphT-I"; // cheia ta
      const url = `https://factchecktools.googleapis.com/v1alpha1/claims:search?query=${encodeURIComponent(query)}&languageCode=en&key=${apiKey}`;

      try {
        const res = await fetch(url);
        const data = await res.json();

        if (!data.claims || data.claims.length === 0) {
          resultsBox.innerHTML = `<em>No verified articles found. Try rephrasing the statement or use a simpler one.</em>`;
          return;
        }

        const html = data.claims.map(claim => {
          const claimText = claim.text || "(No claim text)";
          const reviews = claim.claimReview?.map(r =>
            `<div class="claim-box">
              <div><strong>Claim:</strong> ${claimText}</div>
              <div><strong>Source:</strong> ${r.publisher?.name || "Unknown"}</div>
              <div><strong>Rating:</strong> ${r.textualRating || "N/A"}</div>
              <a href="${r.url}" target="_blank">Read more</a>
            </div>`
          ).join("") || "";
          return reviews;
        }).join("");

        resultsBox.innerHTML = html || "<em>No verified results available.</em>";
      } catch (err) {
        resultsBox.innerHTML = `<em>Error querying Google Fact Check API.</em>`;
        console.error("Fact Check API error:", err);
      }
    });
  });
});
