chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "extractContent") {
    // Heuristic: dacă pagina are titlu + paragrafe -> presupunem că e știre
    const title = document.querySelector("h1")?.innerText || document.title || "";
    const paragraphs = Array.from(document.querySelectorAll("p"))
                            .map(p => p.innerText.trim())
                            .filter(p => p.length > 30); // elimină p-uri irelevante

    const text = paragraphs.join(" ").slice(0, 5000); // limităm dimensiunea

    // Considerăm că e știre dacă sunt suficient de multe paragrafe
    if (paragraphs.length < 3 || text.split(" ").length < 100) {
      sendResponse({ title, text: "", error: "Not a valid news article" });
    } else {
      sendResponse({ title, text });
    }
  }
});
