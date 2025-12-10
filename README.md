# Hermes Logistics AI Assistant

Hermes is a full-stack logistics AI copilot with FastAPI backend and a Bootstrap + Plotly frontend. It supports multiple NLP strategies (rule-based, similarity, logistic regression classifier) and ships with mock shipment data.

## 🎥 Demo Video

Watch the demo video to see Hermes in action:

[![Hermes Logistics AI Assistant Demo](https://img.youtube.com/vi/dNkY6ezW46o/0.jpg)](https://youtu.be/dNkY6ezW46o)

**Direct Link:** https://youtu.be/dNkY6ezW46o

---

## Features

- **Mock Dataset**: `data/shipments.csv` (5,000 rows) auto-generated if missing
- **Three Query Methods**: Switchable via UI or `method` param
  - Rule-based / Keywords (regex pattern matching)
  - Similarity (TF-IDF + cosine similarity)
  - ML Classifier (Logistic Regression)
- **Rich Responses**: Text + chart-ready data (bar, pie, line, prediction charts)
- **Interactive Dashboard**: Plotly charts for routes, warehouses, and delay reasons
- **Time-based Queries**: Support for "last week", "last month", specific months, etc.
- **Advanced Predictions**: 
  - Next week prediction
  - Next month prediction
  - Next year prediction
  - Enhanced ML models with seasonal patterns, lag features, rolling statistics
- **Evaluation Script**: Compare methods by F1-score and latency
- **Dockerized**: Backend (Python 3.10) and frontend (Nginx)

---

## Quickstart

### Using Docker (Recommended):
```bash
docker-compose up --build
```

- **Frontend**: http://localhost/
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Local Development:
```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Open `frontend/index.html` in a browser or serve it via simple HTTP server.

---

## Data Structure

### Shipments Dataset
The system uses a CSV file (`data/shipments.csv`) containing synthetic shipment data:

**Columns:**
- `id`: Unique shipment identifier (1-5000)
- `route`: Route identifier (A-J, 10 routes)
- `warehouse`: Warehouse identifier (WH1-WH4, 4 warehouses)
- `delivery_time`: Delivery time in days (1-10 days)
- `delay_minutes`: Delay in minutes (0-120 minutes)
- `delay_reason`: Reason for delay (Weather, Traffic, Mechanical, None)
- `date`: Shipment date (random dates from 2024-01-01 to 2025-12-10)

**Dataset Size:** 5,000 records  
**Date Range:** January 1, 2024 - December 10, 2025 (current date)

### Example Data:
```csv
id,route,warehouse,delivery_time,delay_minutes,delay_reason,date
1,A,WH1,5,45,Weather,2024-03-15
2,B,WH2,3,0,None,2024-03-16
3,C,WH3,7,89,Traffic,2024-03-17
```

---

## System Architecture

### Architecture Overview
Hermes is a full-stack application with separated frontend and backend, containerized using Docker.

```
┌─────────────────┐
│   Frontend      │  (HTML/CSS/JS)
│   Port: 80      │  - Chat interface
│                 │  - Dashboard with charts
│                 │  - Plotly.js visualizations
└────────┬────────┘
         │ HTTP/REST
         │
┌────────▼────────┐
│   Backend       │  (FastAPI)
│   Port: 8000    │  - Query processing
│                 │  - 3 classification methods
│                 │  - Data aggregation
└────────┬────────┘
         │
┌────────▼────────┐
│   Data Layer    │
│   shipments.csv │  (Pandas DataFrame)
└─────────────────┘
```

### Components:

#### Backend (`backend/app.py`)
- **FastAPI** web framework
- **Pandas** for data loading and manipulation
- **Scikit-learn** for ML classification (TF-IDF, Logistic Regression, Ridge, Random Forest)
- **3 Query Classification Methods:**
  1. Rule-based (regex pattern matching)
  2. Similarity-based (TF-IDF + cosine similarity)
  3. ML Classifier (Logistic Regression)

#### Frontend (`frontend/`)
- **HTML/CSS/JS** (vanilla, no framework)
- **Bootstrap 5** for styling
- **Plotly.js** for interactive charts
- **FontAwesome** for icons
- **Collapsible Dashboard** for better chat space

#### Docker Setup
- `Dockerfile.backend`: Python 3.10, FastAPI, dependencies
- `Dockerfile.frontend`: Nginx to serve static files
- `docker-compose.yml`: Orchestrates both services

---

## Workflow

### Query Processing Flow:

```
User Query
    │
    ▼
Frontend (script.js)
    │
    ├─► Method Selection (1, 2, or 3)
    │
    ▼
POST /api/query
    │
    ├─► Extract time filters (month, last week, etc.)
    ├─► Filter dataset
    │
    ▼
Intent Classification
    │
    ├─► Method 1: Rule-based (regex)
    ├─► Method 2: Similarity (TF-IDF)
    └─► Method 3: ML Classifier (LogReg)
    │
    ▼
Intent Handler
    │
    ├─► delays_by_route → _delays_by_route()
    ├─► top_warehouses → _top_warehouses()
    ├─► delay_reasons → _delay_reasons()
    ├─► predict_delay → _predict_delay_next_week()
    ├─► predict_delay_month → _predict_delay_next_month()
    └─► predict_delay_year → _predict_delay_next_year()
    │
    ▼
Response Generation
    │
    ├─► Text response
    ├─► Table data
    └─► Chart data (Plotly format)
    │
    ▼
Frontend Rendering
    │
    ├─► Display text in chat bubble
    ├─► Render chart (bar/pie/line/prediction)
    └─► Update dashboard
```

### Intent Handlers:

1. **`_delays_by_route()`**: Groups by route, calculates average delay, returns top route
2. **`_top_warehouses()`**: Groups by warehouse, calculates average delivery time, returns top N
3. **`_delay_reasons()`**: Counts delay reasons, returns distribution
4. **`_predict_delay_next_week()`**: Enhanced ML model with seasonal patterns, lag features, rolling statistics
5. **`_predict_delay_next_month()`**: Monthly prediction using enhanced features
6. **`_predict_delay_next_year()`**: Yearly prediction using trend analysis

---

## Query Understanding

### Method 1: Rule-based / Keywords
- Uses regex patterns to match keywords
- Patterns:
  - `"most delay|highest delay|delay.*route"` → `delays_by_route`
  - `"warehouse|processing time|top warehouse"` → `top_warehouses`
  - `"reason|why|cause|breakdown"` → `delay_reasons`
  - `"next week"` → `predict_delay`
  - `"next month"` → `predict_delay_month`
  - `"next year"` → `predict_delay_year`

**Example:**
- Query: "Which route had the most delays?"
- Match: `"most delay"` pattern
- Intent: `delays_by_route`

### Method 2: Similarity (TF-IDF)
- Converts query to TF-IDF vector
- Compares with training templates using cosine similarity
- Selects intent of most similar template

**Training Templates:**
- From `INTENT_TEMPLATES` dictionary
- Includes paraphrases: "Tell me about {intent}"

**Example:**
- Query: "What routes have highest delays?"
- Similar to: "Which route had the most delays?"
- Intent: `delays_by_route`

### Method 3: ML Classifier (Logistic Regression)
- Trained on synthetic data from `INTENT_TEMPLATES`
- Uses TF-IDF vectorization
- 80/20 train/test split
- Predicts intent class directly

**Training Process:**
```python
train_texts, train_labels = generate_synthetic_training()
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, train_labels)
```

---

## Data Summarization

### Aggregation Functions:

1. **Route Delays:**
   - Groups by `route`
   - Calculates `mean(delay_minutes)`
   - Sorts descending
   - Returns top route with average delay

2. **Warehouse Performance:**
   - Groups by `warehouse`
   - Calculates `mean(delivery_time)`
   - Sorts ascending (fastest first)
   - Returns top N warehouses

3. **Delay Reasons:**
   - Counts `value_counts()` of `delay_reason`
   - Returns distribution with percentages

4. **Time-based Filtering:**
   - Month filter: `df[df["date"].dt.month == month]`
   - Last week: `df[df["date"] >= cutoff]` where cutoff = max_date - 7 days
   - Last month: `df[df["date"] >= cutoff]` where cutoff = max_date - 1 month

### Response Format:
```json
{
  "method_used": "rule_based|similarity|ml_classifier",
  "intent": "delays_by_route|top_warehouses|delay_reasons|predict_delay|predict_delay_month|predict_delay_year",
  "text": "Human-readable response",
  "table_data": [{"metric": "value", ...}],
  "chart_data": {
    "type": "bar|pie|line|prediction",
    "x": [...],
    "y": [...],
    "title": "..."
  }
}
```

---

## Example Queries and Outputs

### Example 1: Route Delays
**Query:** "Which route had the most delays last week?"

**Response:**
- **Text:** "Route D has the highest average delay at 61.5 minutes in last week."
- **Chart:** Bar chart showing average delay by route (A-J)
- **Table:** List of routes with average delays

### Example 2: Warehouse Performance
**Query:** "Top 3 warehouses by processing time"

**Response:**
- **Text:** "Top 3 warehouses by fastest delivery: WH3, WH1, WH2."
- **Chart:** Bar chart showing average delivery time by warehouse
- **Table:** List of top warehouses with delivery times

### Example 3: Delay Reasons
**Query:** "Show total delayed shipments by delay reason"

**Response:**
- **Text:** "Total delay reasons breakdown: 5000 shipments across 4 categories. Top reason: Weather with 1281 occurrences (25.6%)."
- **Chart:** Pie chart showing distribution (Weather, Traffic, Mechanical, None)
- **Table:** Count for each reason

### Example 4: Prediction - Next Week
**Query:** "Predict delay next week"

**Response:**
- **Text:** "Predicted average delay next week: 60.1 minutes. Using enhanced ML model with advanced features (seasonal patterns, lag features, rolling statistics). Model accuracy (R²): 40.91%."
- **Chart:** Multi-trace chart:
  - Historical daily delays (scatter)
  - Weekly average trend (line)
  - Prediction point (diamond marker)
- **Table:** Prediction details, model method, features, R² score (if >= 10%)

### Example 5: Prediction - Next Month
**Query:** "Predict the delay rate for next month"

**Response:**
- **Text:** "Predicted average delay for January 2026: 63.0 minutes. Using enhanced ML model with advanced features (seasonal patterns, lag features, rolling statistics). Model accuracy (R²): 99.87%."
- **Chart:** Monthly prediction chart with historical data and trend
- **Table:** Monthly prediction details

### Example 6: Time-based Filtering
**Query:** "What was the average delay in October?"

**Response:**
- **Text:** "Route A has the highest average delay at 65.7 minutes in October."
- **Chart:** Bar chart filtered to October data only
- **Table:** Routes with October delays

---

## API Endpoints

- `POST /api/query` - Body: `{"query": "...", "method": 1|2|3, "context": {...}}`
- `GET /api/query?q=...&method=1|2|3` - Query via GET
- `GET /api/stats` - Dashboard aggregates
- `GET /health` - Health check

---

## Evaluation

### Running Evaluation:
```bash
# Inside backend container
docker-compose exec backend python eval.py

# Or locally (if dependencies installed)
python eval.py
```

### Evaluation Metrics:

**Accuracy (F1-Score):**
- Measures classification accuracy across all intent classes
- Formula: `F1 = 2 * (precision * recall) / (precision + recall)`
- Evaluated on 50 test queries with ground truth labels

**Response Time:**
- Average query processing time (seconds)
- Includes: classification + data filtering + aggregation + response generation

**Expected Output:**
```
Evaluating 3 methods on 50 queries...
Method 1 (Rule-based): F1 0.XXX, Avg Time 0.XXXXs
Method 2 (Similarity): F1 0.XXX, Avg Time 0.XXXXs
Method 3 (ML Classifier): F1 0.XXX, Avg Time 0.XXXXs
```

### Explainability

**Data Source Transparency:**
- All responses include the method used (`method_used`)
- Chart data shows exact values used
- Table data provides detailed breakdown

**Model Explainability:**
- **Rule-based:** Shows which pattern matched
- **Similarity:** Can show most similar template
- **ML Classifier:** Shows predicted intent class

**Response Justification:**
- Text responses explain the calculation
- Charts visualize the underlying data
- Tables show raw numbers

---

## Technical Details

### Dependencies:
- **Backend:** fastapi, uvicorn, pandas, scikit-learn, numpy, seaborn
- **Frontend:** Bootstrap 5, Plotly.js, FontAwesome
- **Python:** 3.10+
- **Docker:** Docker Compose

### File Structure:
```
Hermes/
├── backend/
│   └── app.py              # FastAPI application
├── frontend/
│   ├── index.html          # Main HTML
│   ├── style.css           # Styling
│   └── script.js           # Frontend logic
├── data/
│   └── shipments.csv       # Dataset (5000 rows)
├── eval.py                 # Evaluation script
├── generate_data.py        # Data generation script
├── requirements.txt        # Python dependencies
├── Dockerfile.backend      # Backend container
├── Dockerfile.frontend     # Frontend container
├── docker-compose.yml      # Docker orchestration
└── README.md               # This file
```

---

## Features Highlights

### Enhanced Prediction Models
- **Multiple Models**: Automatically selects best model (Ridge, Linear Regression, Random Forest)
- **Advanced Features**: 
  - Seasonal patterns (sin/cos encoding)
  - Lag features (previous period values)
  - Rolling statistics (3-period and 6-period averages)
  - Quarter, day of year features
- **High Accuracy**: R² scores up to 99%+ for monthly predictions

### Context-Aware Queries
- **Follow-up Support**: "how about last month?" remembers previous intent
- **Time Filtering**: Automatic extraction of time periods (last week, last month, specific months)
- **Smart Classification**: Handles variations and paraphrases

### Interactive Dashboard
- **Collapsible**: Click chevron button to expand/collapse
- **Auto-expand Chat**: Chatbox expands when dashboard is collapsed
- **Real-time Updates**: Refresh dashboard button to reload stats

---

## Future Improvements

1. **Enhanced ML Model:**
   - More training data
   - Fine-tuning hyperparameters
   - Ensemble methods

2. **Additional Features:**
   - User authentication
   - Query history
   - Export results
   - Real-time data updates

3. **Performance:**
   - Caching frequently accessed data
   - Database instead of CSV
   - API rate limiting

4. **Explainability:**
   - Show confidence scores
   - Highlight matched patterns
   - Display feature importance

---
## License

This project is provided as-is for educational and demonstration purposes.

---

**Version:** 1.0.0  
**Last Updated:** December 2024
