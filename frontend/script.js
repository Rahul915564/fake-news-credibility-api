async function checkNews() {
  const title = document.getElementById("title").value.trim();
  const text = document.getElementById("text").value.trim();
  const result = document.getElementById("result");

  if (!title || !text) {
    result.innerHTML = "<span style='color:orange'>Please enter both title and text.</span>";
    return;
  }

  result.innerHTML = "⏳ Analyzing...";

  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text })
    });

    const data = await res.json();

    // Color by label
    const color = data.label === "REAL" ? "#22c55e" : "#ef4444";

    // Risk badge color
    let riskColor = "#22c55e";
    if (data.risk === "Medium Risk") riskColor = "#f59e0b";
    if (data.risk === "High Risk") riskColor = "#ef4444";

    result.innerHTML = `
      <div style="font-size:22px;font-weight:bold;color:${color}">
        ${data.label}
      </div>

      <div style="margin-top:6px">
        Confidence: <b>${data.confidence}%</b>
      </div>

      <div style="margin-top:6px;color:${riskColor};font-weight:bold">
        ${data.risk}
      </div>

      <hr style="margin:12px 0"/>

      <div style="text-align:left">
        <b>Why this result?</b>
        <ul>
          ${data.top_words.map(w => `<li>${w}</li>`).join("")}
        </ul>
      </div>
    `;
  } catch (e) {
    result.innerHTML = "<span style='color:red'>❌ Backend not reachable</span>";
  }
}
