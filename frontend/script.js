const messagesBox = document.getElementById("messages");
const input = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const loadStatsBtn = document.getElementById("load-stats");

const backendBase =
  window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : window.location.origin.replace(/:\\d+$/, ":8000");

// Context tracking for follow-up questions
let lastContext = {
  intent: null,
  originalQuery: null
};

// Counter for unique chart IDs
let chartCounter = 0;

function appendMessage(text, author = "bot") {
  if (!messagesBox) {
    console.error("Messages box not found!");
    return;
  }
  const wrapper = document.createElement("div");
  wrapper.className = `message ${author}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  messagesBox.appendChild(wrapper);
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

function getMethod() {
  const selected = document.querySelector("input[name='method']:checked");
  return selected ? Number(selected.value) : 1;
}

function isFollowUpQuery(query) {
  // Detect follow-up questions that should use previous context
  const followUpPatterns = [
    /^how about/i,
    /^what about/i,
    /^and/i,
    /^also/i,
    /^then/i,
    /^next/i,
    /^last (week|month|year)/i,
    /^this (week|month|year)/i
  ];
  return followUpPatterns.some(pattern => pattern.test(query.trim()));
}

async function sendQuery() {
  const query = input.value.trim();
  if (!query) return;
  appendMessage(query, "user");
  input.value = "";
  appendMessage("Thinking...", "bot");

  try {
    // For follow-up queries, use context instead of enhancing query
    const payload = { 
      query: query,  // Use original query, not enhanced
      method: getMethod()
    };
    
    // Add context if we have a previous intent (for follow-up questions)
    if (lastContext.intent && isFollowUpQuery(query)) {
      payload.context = { previous_intent: lastContext.intent };
    }

    const res = await fetch(`${backendBase}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    const data = await res.json();
    
    // Remove "Thinking..." message safely
    if (messagesBox && messagesBox.lastChild) {
      messagesBox.lastChild.remove();
    }
    
    if (!data || !data.text) {
      throw new Error("Invalid response from server");
    }
    
    appendMessage(data.text, "bot");
    if (data.chart_data) {
      renderChatChart(data.chart_data);
    }
    
    // Update context for next query
    if (data.intent) {
      lastContext.intent = data.intent;
      lastContext.originalQuery = query; // Store original, not enhanced
    }
  } catch (err) {
    console.error("Query error:", err);
    // Remove "Thinking..." message safely
    if (messagesBox && messagesBox.lastChild) {
      messagesBox.lastChild.remove();
    }
    appendMessage("Error: Unable to process query. Please try again.", "bot");
  }
}

// Pastel color palette inspired by seaborn
const pastelColors = [
  "#A8D5E2", // pastel blue
  "#F4A5AE", // pastel pink
  "#B8E6B8", // pastel green
  "#FFE5B4", // pastel yellow
  "#D4A5FF", // pastel purple
  "#FFD4A5", // pastel orange
  "#B8E6D1", // pastel mint
  "#FFB3BA", // pastel coral
];

function renderChatChart(chart) {
  if (!chart || !chart.type) return;
  // Create unique ID for each chart to avoid conflicts
  const targetId = `chart-chat-${chartCounter++}`;
  const holder = document.createElement("div");
  holder.id = targetId;
  holder.className = "plot mt-3";
  messagesBox.appendChild(holder);
  const common = { 
    title: chart.title || "Result",
    font: { family: "Arial, sans-serif", size: 12 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)"
  };
  if (chart.type === "bar") {
    Plotly.react(
      holder,
      [{ 
        x: chart.x, 
        y: chart.y, 
        type: "bar", 
        marker: { 
          color: pastelColors[0],
          line: { color: "#ffffff", width: 1 }
        } 
      }],
      { ...common, margin: { t: 40, b: 40, l: 50, r: 20 } }
    );
  } else if (chart.type === "pie") {
    Plotly.react(
      holder,
      [{ 
        labels: chart.labels, 
        values: chart.values, 
        type: "pie",
        marker: {
          colors: pastelColors.slice(0, chart.labels.length),
          line: { color: "#ffffff", width: 2 }
        },
        textinfo: "label+percent",
        textposition: "outside"
      }],
      { ...common, margin: { t: 40, b: 40, l: 40, r: 40 } }
    );
  } else if (chart.type === "line") {
    Plotly.react(
      holder,
      [{ 
        x: chart.x, 
        y: chart.y, 
        type: "scatter", 
        mode: "lines+markers",
        marker: { color: pastelColors[0], size: 6 },
        line: { color: pastelColors[0], width: 2 }
      }],
      { ...common, margin: { t: 40, b: 40, l: 50, r: 20 } }
    );
  } else if (chart.type === "prediction") {
    // Multi-trace chart for prediction: historical data, trend, and prediction point
    const traces = [];
    
    if (chart.historical) {
      traces.push({
        x: chart.historical.x,
        y: chart.historical.y,
        type: "scatter",
        mode: chart.historical.mode || "markers",
        name: chart.historical.name || "Historical",
        marker: chart.historical.marker || { color: pastelColors[0], size: 4, opacity: 0.6 },
        showlegend: true
      });
    }
    
    if (chart.trend) {
      traces.push({
        x: chart.trend.x,
        y: chart.trend.y,
        type: "scatter",
        mode: chart.trend.mode || "lines+markers",
        name: chart.trend.name || "Trend",
        line: chart.trend.line || { color: pastelColors[1], width: 2 },
        marker: { color: pastelColors[1], size: 5 },
        showlegend: true
      });
    }
    
    if (chart.prediction) {
      traces.push({
        x: chart.prediction.x,
        y: chart.prediction.y,
        type: "scatter",
        mode: chart.prediction.mode || "markers",
        name: chart.prediction.name || "Prediction",
        marker: chart.prediction.marker || { 
          color: pastelColors[4], 
          size: 12, 
          symbol: "diamond",
          line: { color: "#ffffff", width: 2 }
        },
        showlegend: true
      });
    }
    
    const layout = {
      ...common,
      title: chart.title || "Prediction Chart",
      margin: { t: 50, b: 50, l: 60, r: 30 },
      xaxis: { title: "Date", type: "date" },
      yaxis: { title: "Delay (minutes)" },
      legend: { x: 0, y: 1, bgcolor: "rgba(255,255,255,0.8)" },
      annotations: (chart.r2_score !== undefined && chart.r2_score !== null) ? [{
        x: 0.02,
        y: 0.98,
        xref: "paper",
        yref: "paper",
        text: `Model R²: ${(chart.r2_score * 100).toFixed(1)}%`,
        showarrow: false,
        bgcolor: "rgba(255,255,255,0.8)",
        bordercolor: pastelColors[4],
        borderwidth: 1
      }] : []
    };
    
    Plotly.react(holder, traces, layout);
  }
}

async function loadDashboard() {
  try {
    const res = await fetch(`${backendBase}/api/stats`);
    const data = await res.json();
    renderCardChart("chart-routes", data.routes.chart);
    renderCardChart("chart-warehouses", data.warehouses.chart);
    renderCardChart("chart-reasons", data.reasons.chart);
  } catch (err) {
    console.error(err);
    appendMessage("Unable to load dashboard stats.", "bot");
  }
}

function renderCardChart(id, chart) {
  if (!chart) return;
  const common = { 
    title: chart.title || "", 
    margin: { t: 30, b: 30, l: 40, r: 20 },
    font: { family: "Arial, sans-serif", size: 11 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)"
  };
  if (chart.type === "bar") {
    Plotly.react(
      id,
      [{ 
        x: chart.x, 
        y: chart.y, 
        type: "bar", 
        marker: { 
          color: pastelColors[0],
          line: { color: "#ffffff", width: 1 }
        } 
      }],
      common
    );
  } else if (chart.type === "pie") {
    Plotly.react(
      id,
      [{ 
        labels: chart.labels, 
        values: chart.values, 
        type: "pie",
        marker: {
          colors: pastelColors.slice(0, chart.labels.length),
          line: { color: "#ffffff", width: 2 }
        },
        textinfo: "label+percent",
        textposition: "outside"
      }],
      common
    );
  } else if (chart.type === "line") {
    Plotly.react(
      id,
      [{ 
        x: chart.x, 
        y: chart.y, 
        type: "scatter", 
        mode: "lines+markers",
        marker: { color: pastelColors[0], size: 5 },
        line: { color: pastelColors[0], width: 2 }
      }],
      common
    );
  }
}

// Dashboard toggle functionality
const toggleDashboardBtn = document.getElementById("toggle-dashboard");
const dashboardBody = document.getElementById("dashboard-body");
const dashboardCard = document.getElementById("dashboard-card");
const dashboardIcon = document.getElementById("dashboard-icon");
const chatCard = document.getElementById("chat-card");

let dashboardExpanded = true;

if (toggleDashboardBtn && dashboardBody && dashboardIcon && dashboardCard) {
  toggleDashboardBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    dashboardExpanded = !dashboardExpanded;
    
    if (dashboardExpanded) {
      // Expand dashboard
      dashboardBody.classList.remove("collapsed");
      dashboardCard.classList.remove("collapsed");
      dashboardIcon.classList.remove("rotated");
      messagesBox.classList.remove("expanded");
      if (chatCard) chatCard.classList.remove("expanded");
    } else {
      // Collapse dashboard - hide body, expand chat
      dashboardBody.classList.add("collapsed");
      dashboardCard.classList.add("collapsed");
      dashboardIcon.classList.add("rotated");
      messagesBox.classList.add("expanded");
      if (chatCard) chatCard.classList.add("expanded");
    }
  });
}

sendBtn.addEventListener("click", sendQuery);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendQuery();
});
loadStatsBtn.addEventListener("click", loadDashboard);

appendMessage("Hello! Ask me about logistics performance or delays.", "bot");
loadDashboard();


