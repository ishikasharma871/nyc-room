# 🏙️ NYC Airbnb Room Type Predictor

**Guess whether an Airbnb listing in New York City is an entire home, a private room, or a shared room — just from its price, location, and booking details.**

🔗 **Live demo:** open `index.html` (or the deployed link, if you've set one up) and try it in your browser — no coding needed.
📓 **Full analysis:** [`nyc_airbnb_room_type_classification.ipynb`](./nyc_airbnb_room_type_classification.ipynb)
🎢 **Visual walkthrough of how it all fits together:** [`the_build_line_guide.html`](./the_build_line_guide.html) — open this in a browser for a fun, animated "subway map" explanation of the whole project.

---

## 🤔 What is this project, in plain English?

Every Airbnb listing in New York City is one of three types:

- 🏠 **Entire home/apartment** – you get the whole place to yourself
- 🚪 **Private room** – you get your own room inside someone else's home
- 🛏️ **Shared room** – you share the room itself with other guests

This project builds a **machine learning model** — a computer program that learns patterns from past examples — that looks at a listing's details (like its price, neighbourhood, how many nights you must book, how many reviews it has, etc.) and **predicts which of the three room types it most likely is.**

Think of it like this: if you showed a real estate expert a listing's price, location, and booking history — but hid the actual photos and title — could they guess what kind of room it is? That's exactly what this model learned to do, using real historical data from ~49,000 NYC Airbnb listings.

It doesn't just predict the room type — it also tells you **how confident** it is (e.g. "85% sure this is a Private Room").

---

## 🧭 How the whole project fits together

This project isn't just "a notebook that makes a prediction." It's a complete, real pipeline — the same kind used to ship machine learning to actual users. There are 5 stages, like stops on a subway line:

| Stop | What happens | Where it lives |
|---|---|---|
| 🧠 **1. Model** | Clean the data, explore it, engineer features, train and compare several ML algorithms, pick the best one | `nyc_airbnb_room_type_classification.ipynb` |
| 🧊 **2. Freeze it** | Save the finished, trained model as a single reusable file so it never has to be retrained | `Model_Pipeline.pkl` (produced by the notebook) |
| ⚡ **3. Give it a voice (API)** | Wrap the frozen model in a small web server that can accept a listing's details and reply with a prediction | `main.py` (built with FastAPI) |
| 🎨 **4. Give it a face (UI)** | A simple, friendly web page where anyone can type in listing details and see the prediction, no coding required | `index.html`, `style.css`, `script.js` |
| 🚀 **5. Go live** | Deploy the API and the web page so anyone in the world can use it, not just on one laptop | Render (see [Deployment](#-deployment)) |

If you want the fun, illustrated version of this table, open **`the_build_line_guide.html`** in a browser — it walks through each stop with an interactive animation.

---

## 🔬 Step 1: The Machine Learning model (in plain English)

All the "thinking" work happens in the Jupyter notebook: `nyc_airbnb_room_type_classification.ipynb`. Here's what it does, step by step:

1. **Load the data** — ~49,000 real NYC Airbnb listings from 2019 ([Kaggle dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)), including price, location, reviews, and more.
2. **Explore the data (EDA)** — look at how prices are distributed, which boroughs have the most listings, whether room types are balanced (they're not — shared rooms are rare), and how features relate to each other.
3. **Clean the data** — drop columns that don't help (like listing ID or host name), fill in missing values sensibly (e.g. no reviews yet = 0 reviews per month), and cap extreme outliers (like a $10,000/night mistake) so they don't confuse the model.
4. **Split the data** — set aside a chunk of listings the model will *never see* during training, so we can honestly test it later, like a final exam.
5. **Preprocess consistently** — numbers get scaled, categories (like borough names) get converted into a format the model understands, all done through a repeatable pipeline so there's no cheating or data leakage.
6. **Try multiple algorithms** — Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting were all tested and compared fairly using cross-validation.
7. **Pick the winner and tune it** — Random Forest performed best, so its settings (like how many trees to use) were fine-tuned to squeeze out extra performance.
8. **Test it honestly** — the tuned model was scored on the untouched test data for a realistic sense of real-world performance.
9. **Save it** — the entire trained pipeline (cleaning + prediction, all in one) was frozen into a single file, `Model_Pipeline.pkl`, ready to be reused instantly without retraining.

### 📊 How good is the model?

| Model tried | Accuracy | F1 Score (macro) |
|---|---|---|
| Logistic Regression | 65.9% | 0.522 |
| Decision Tree | 78.2% | 0.647 |
| Gradient Boosting | 85.0% | 0.705 |
| **Random Forest (winner)** | **85.1%** | **0.715** |

After tuning, the final **Random Forest** model reached:
- **~85.6% accuracy** on completely unseen listings
- **F1 score of ~0.74** (a fairer measure than accuracy alone, since "Shared Room" listings are much rarer than the other two types — a model could get high accuracy just by ignoring them, but F1 catches that)

**Plain-English takeaway:** if you gave this model 100 listings it had never seen before, it would correctly guess the room type for about 85 to 86 of them.

---

## ⚡ Step 2: The API (`main.py`)

A trained model sitting in a notebook is only useful to the person who wrote it. The **API** turns it into something *any* app, website, or script can talk to.

Built with **[FastAPI](https://fastapi.tiangolo.com/)**, `main.py`:
- Loads the frozen model (`Model_Pipeline.pkl`) once, when the server starts
- Defines exactly what information it needs about a listing (with built-in validation — e.g. latitude must be a real latitude, price must be positive)
- Exposes an endpoint, `/predict`, that accepts those details and returns:
  - the predicted room type, and
  - the model's confidence for each possible room type

### Try it yourself (once running)

Send a `POST` request to `/predict` with a JSON body like:

```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "price": 150,
  "minimum_nights": 3,
  "number_of_reviews": 25,
  "reviews_per_month": 1.2,
  "calculated_host_listings_count": 1,
  "availability_365": 180,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Chelsea"
}
```

And you'll get back something like:

```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.78, 0.19, 0.03]
}
```

---

## 🎨 Step 3: The web interface

Most people don't want to send raw JSON requests — they want to click and type. That's what `index.html`, `style.css`, and `script.js` are for: a clean, NYC-skyline-themed form where you fill in a listing's details and instantly see the predicted room type, styled and animated for a pleasant experience. Under the hood, it simply calls the same `/predict` API described above.

---

## 🚀 Deployment

The project ships as two separate pieces, each deployed to **[Render](https://render.com/)**:

- The **web interface** (`index.html`, `style.css`, `script.js`) is deployed as a **Static Site**.
- The **API** (`main.py` + the frozen model) is deployed as a **Web Service**.

The interface's JavaScript is already pointed at the live API URL, so once both pieces are deployed, the whole thing works end-to-end for anyone with the link — no setup required on their end.

---

## 🗂️ Project structure

```
nyc-room/
├── nyc_airbnb_room_type_classification.ipynb   # Full ML workflow: EDA → cleaning → training → evaluation
├── main.py                                     # FastAPI backend that serves predictions
├── requirements.txt                            # Python packages needed to run the API
├── runtime.txt                                 # Python version used for deployment
├── index.html                                  # The web page / user interface
├── style.css                                   # Styling for the web page
├── script.js                                   # Connects the web page to the API
├── the_build_line_guide.html                   # An animated, illustrated tour of the whole project
└── README.md                                   # You are here
```

> Note: `Model_Pipeline.pkl` (the saved, trained model) is produced by running the notebook. If it isn't already in the repo, run the notebook once to generate it before starting the API.

---

## 🛠️ Tech stack

| Purpose | Tool |
|---|---|
| Data analysis & modeling | Python, pandas, NumPy, scikit-learn |
| Visualization (in the notebook) | Matplotlib, Seaborn |
| Model serving | FastAPI, Pydantic (input validation), Uvicorn |
| Saving/loading the trained model | joblib |
| Frontend | HTML, CSS, JavaScript |
| Hosting | Render |

---

## ▶️ Running it locally

**1. Install the requirements:**
```bash
pip install -r requirements.txt
```

**2. Make sure `Model_Pipeline.pkl` exists** (generated by running the notebook end-to-end, if not already present).

**3. Start the API:**
```bash
uvicorn main:app --reload
```
This starts the server at `http://127.0.0.1:8000`. You can explore and test the `/predict` endpoint interactively at `http://127.0.0.1:8000/docs`.

**4. Open the interface:**
Update the API address inside `script.js` to point at `http://127.0.0.1:8000` if you're testing locally, then simply open `index.html` in your browser.

---

## 💡 What this project demonstrates

- Framing a real-world question as a machine learning classification problem
- A complete, leak-free data cleaning and preprocessing pipeline
- Fairly comparing multiple algorithms instead of picking one arbitrarily
- Hyperparameter tuning and honest evaluation on unseen data
- Turning a trained model into a real, usable web service (not just a notebook)
- Building a simple front end so non-technical users can interact with the model
- Deploying a full-stack ML project so it's accessible to anyone, not just on one machine

---

## 📚 Dataset credit

[New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — via Kaggle.
