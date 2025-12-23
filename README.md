# TS-Arena: A Pre-registered Live-Data Forecasting Platform 🏟️

Time Series Foundation Models (TSFMs) represent a significant advancement in forecasting capabilities. However, the rapid scaling of these models triggers an evaluation crisis characterized by information leakage and performance reporting issues. Traditional benchmarks are often compromised because TSFMs may inadvertently train on datasets later used for evaluation.

To address these challenges, we introduce **TS-Arena**, a platform for live-data forecasting. The platform reframes the test set as the yet-unseen future, ensuring that evaluation data does not exist at the time of model prediction.

## The Concept of Pre-registration 📝

The core of our methodology is the pre-registration of forecasts. This mechanism requires that a prediction is irrevocably committed at a specific time point  before the ground truth observations manifest. By enforcing this strictly causal timeline, we mitigate two primary forms of information leakage:

* 
**Test Set Contamination**: This occurs when benchmark data is exposed to a model during its pre-training phase. Since our platform uses real-time future data, the target values cannot be part of any training corpus.


* 
**Global Pattern Memorization**: Models can exploit shared global shocks, such as economic crises, that influence many series simultaneously. A global time-split at  ensures models rely on learned dynamics rather than recognizing events they have already seen in other series during training.



## Live Challenges and Visualization 🌐

You can view the rolling leaderboards and active challenges live on Huggingface:
👉 **[TS-Arena on Huggingface](https://huggingface.co/spaces/DAG-UPB/TS-Arena)** 

---

## System Architecture 🏗️

The TS-Arena ecosystem is distributed across three specialized repositories to manage data, models, and user interaction.

### 1. TS-Arena Backend

The [Backend Infrastructure](https://github.com/DAG-UPB/ts-arena-backend) powers the platform by orchestrating challenges and managing data provenance. It consists of several microservices:

* **Data Portal**: Responsible for fetching ground truth data from external providers like the U.S. Energy Information Administration (EIA) and **SMARD** (Bundesnetzagentur).


* 
**API Portal**: Handles model registration, accepts incoming forecasts, and manages the evaluation process.


* **Dashboard API**: Serves the frontend by retrieving statistics and leaderboard data.

### 2. TS-Arena Models

The [Models Repository](https://github.com/DAG-UPB/ts-arena-models) contains the implementation of various state-of-the-art forecasting models. These models serve as baseline participants in the challenges:

* 
**Foundation Models**: Includes Chronos, TimesFM, Moirai, MOMENT, and Time-MoE.


* 
**Standard Baselines**: Includes statistical methods and deep learning models like NHITS or PatchTST.
The repository provides a containerized environment to ensure context parity and full reproducibility across all implemented models.



### 3. TS-Arena Frontend

The [Frontend Dashboard](https://github.com/DAG-UPB/ts-arena-frontend) is built with Streamlit to provide an interactive interface for the benchmark. It allows users to:

* Filter model rankings based on performance metrics like MASE.


* Visualize active and completed challenges using interactive Plotly charts.


* Access information on how to participate and register new models.



## Participation 🤝

The platform is designed to be inclusive for both academic and industrial researchers. Participants can join through containerized inference for maximum rigor or the **Bring Your Own Prediction (BYOP)** mode for proprietary models.