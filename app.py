
import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Battery Prediction System",
    page_icon="🔋",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results"

st.title("🔋 AI-Enabled Battery Prediction System")

st.markdown(
    """
    **Current, Voltage and Temperature-Aware Battery Monitoring**

    This prototype uses recorded battery test data to predict
    the next current and voltage readings. Temperature is shown
    as a constant 23°C reference because the original dataset
    contains only one actual temperature measurement.
    """
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    dataset = pd.read_csv(
        RESULTS / "battery_prediction_dataset.csv"
    )

    metrics = pd.read_csv(
        RESULTS / "model_comparison_metrics.csv"
    )

    current_predictions = pd.read_csv(
        RESULTS / "current_test_predictions.csv"
    )

    voltage_predictions = pd.read_csv(
        RESULTS / "voltage_test_predictions.csv"
    )

    return dataset, metrics, current_predictions, voltage_predictions


try:
    df, metrics, current_predictions, voltage_predictions = load_data()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# --------------------------------------------------
# GRAPH DISPLAY FUNCTION
# IMPORTANT: DEFINE THIS BEFORE PAGE NAVIGATION
# --------------------------------------------------

def display_graphs(target):

    st.subheader(f"{target} Actual vs Predicted")

    algorithms = [
        "Decision_Tree",
        "Random_Forest",
        "Gradient_Boosting",
        "SVR",
        "Persistence_Baseline"
    ]

    for algorithm in algorithms:

        filename = f"{target}_{algorithm}_actual_vs_predicted.png"
        image_path = RESULTS / filename

        if image_path.exists():

            st.markdown(
                f"#### {algorithm.replace('_', ' ')}"
            )

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:
            st.warning(f"Graph not found: {filename}")


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Dataset Information",
        "Current Prediction Results",
        "Voltage Prediction Results",
        "Algorithm Comparison",
        "Temperature Monitoring",
        "Interactive Prediction"
    ]
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Available Records", f"{len(df):,}")
    col2.metric("Battery Steps", df["Pgm step"].nunique())
    col3.metric("Test Records", f"{len(current_predictions):,}")
    col4.metric("Reference Temperature", "23 °C")

    st.divider()

    st.subheader("Project Overview")

    st.write(
        """
        This project applies four machine-learning algorithms
        to battery measurement data for next-reading prediction.

        Implemented prediction targets:

        - Current (A)
        - Voltage (V)

        Temperature is retained as a 23°C reference condition.
        """
    )

    st.subheader("Machine Learning Algorithms")

    st.write(
        """
        1. Decision Tree Regression
        2. Random Forest Regression
        3. Gradient Boosting Regression
        4. Support Vector Regression (SVR)

        Persistence Baseline is used as a benchmark.
        """
    )

    st.subheader("Project Workflow")

    st.code(
        """
Recorded Battery Dataset
          |
          v
Data Preprocessing
          |
          v
Feature Preparation
          |
          v
Machine Learning Models
          |
          v
Current and Voltage Prediction
          |
          v
Performance Evaluation
          |
          v
Graphs and Dashboard
        """,
        language="text"
    )


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

elif page == "Dataset Information":

    st.header("Dataset Information")

    st.write("Prepared battery prediction dataset.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Program Steps", df["Pgm step"].nunique())
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Program Step Distribution")

    step_counts = df["Pgm step"].value_counts().sort_index()
    st.bar_chart(step_counts)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Dataset Features")

    st.write(
        """
        - Duration (sec): Recorded test time.
        - Pgm step: Battery test program stage.
        - Current_A: Measured current in amperes.
        - Voltage_V: Measured voltage in volts.
        - Temperature_C: Constant 23°C reference.
        - Next_Current_A: Next current reading.
        - Next_Voltage_V: Next voltage reading.
        """
    )


# --------------------------------------------------
# CURRENT RESULTS
# --------------------------------------------------

elif page == "Current Prediction Results":

    st.header("Current Prediction Results")

    current_metrics = metrics[
        metrics["Target"] == "Current"
    ]

    st.subheader("Current Model Performance")

    st.dataframe(
        current_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Current Prediction Graphs")

    display_graphs("Current")

    st.subheader("Current Test Predictions")

    st.dataframe(
        current_predictions.head(100),
        use_container_width=True
    )


# --------------------------------------------------
# VOLTAGE RESULTS
# --------------------------------------------------

elif page == "Voltage Prediction Results":

    st.header("Voltage Prediction Results")

    voltage_metrics = metrics[
        metrics["Target"] == "Voltage"
    ]

    st.subheader("Voltage Model Performance")

    st.dataframe(
        voltage_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Voltage Prediction Graphs")

    display_graphs("Voltage")

    st.subheader("Voltage Test Predictions")

    st.dataframe(
        voltage_predictions.head(100),
        use_container_width=True
    )


# --------------------------------------------------
# ALGORITHM COMPARISON
# --------------------------------------------------

elif page == "Algorithm Comparison":

    st.header("Algorithm Performance Comparison")

    target = st.selectbox(
        "Choose Prediction Target",
        ["Current", "Voltage"]
    )

    selected_metrics = metrics[
        metrics["Target"] == target
    ]

    st.subheader(f"{target} Performance Metrics")

    st.dataframe(
        selected_metrics,
        use_container_width=True,
        hide_index=True
    )

    for metric in ["MAE", "RMSE", "R2"]:

        st.subheader(f"{target} - {metric} Comparison")

        filename = f"{target}_{metric}_comparison.png"
        image_path = RESULTS / filename

        if image_path.exists():

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:
            st.warning(f"Graph not found: {filename}")

    st.info(
        """
        Lower MAE and RMSE indicate smaller prediction errors.
        Higher R² indicates a better fit to the observed target.
        """
    )


# --------------------------------------------------
# TEMPERATURE MONITORING
# --------------------------------------------------

elif page == "Temperature Monitoring":

    st.header("Temperature Monitoring")

    st.metric("Temperature Reference", "23 °C")

    st.subheader("Temperature Data Status")

    st.warning(
        """
        The original Excel dataset contains only one recorded
        temperature value: 23°C. The remaining temperature
        entries are missing.

        The project uses 23°C as a constant reference condition.
        It does not represent 169,404 independent temperature
        measurements.
        """
    )

    st.subheader("Temperature-Aware Project Scope")

    st.write(
        """
        Temperature is included in the project as an operating
        reference condition.

        Genuine temperature forecasting is not currently
        implemented because the dataset does not contain a
        varying temperature time series.

        Future work can include:
        - Continuous temperature sensor measurements.
        - Temperature prediction using time-series data.
        - Thermal anomaly detection.
        - Adaptive battery control based on temperature.
        """
    )


# --------------------------------------------------
# INTERACTIVE PREDICTION
# --------------------------------------------------

elif page == "Interactive Prediction":

    st.header("Interactive Next-Reading Prediction")

    st.info(
        """
        Enter values in the same units and ranges as the dataset.
        Predictions are based on models trained using recorded data.
        They are not live sensor readings.
        """
    )

    features = [
        "Duration (sec)",
        "Pgm step",
        "Current_A",
        "Voltage_V"
    ]

    @st.cache_resource
    def train_prediction_models():

        training_data = []

        for step_value, group in df.groupby("Pgm step"):

            group = group.sort_values(
                "Duration (sec)"
            ).reset_index(drop=True)

            split = int(len(group) * 0.80)

            training_data.append(group.iloc[:split])

        train_df = pd.concat(
            training_data,
            ignore_index=True
        )

        X_train = train_df[features]

        models = {}

        model_definitions = {

            "Decision Tree": DecisionTreeRegressor(
                max_depth=20,
                min_samples_leaf=5,
                random_state=42
            ),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42
            ),

            "SVR": Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(
                    kernel="rbf",
                    C=10,
                    epsilon=0.001,
                    gamma="scale"
                ))
            ])
        }

        for target_name, target_column in {
            "Current": "Next_Current_A",
            "Voltage": "Next_Voltage_V"
        }.items():

            models[target_name] = {}

            for model_name, model in model_definitions.items():

                if model_name == "SVR" and len(train_df) > 5000:

                    sample = train_df.sample(
                        n=5000,
                        random_state=42
                    )

                    model.fit(
                        sample[features],
                        sample[target_column]
                    )

                else:

                    model.fit(
                        X_train,
                        train_df[target_column]
                    )

                models[target_name][model_name] = model

        return models

    model_choice = st.selectbox(
        "Select Machine Learning Algorithm",
        [
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "SVR"
        ]
    )

    st.subheader("Enter Battery Measurements")

    col1, col2 = st.columns(2)

    with col1:

        duration = st.number_input(
            "Duration (seconds)",
            min_value=9,
            max_value=169436,
            value=10000
        )

        step = st.selectbox(
            "Program Step",
            [1, 2, 3, 4, 5, 6]
        )

    with col2:

        current = st.number_input(
            "Current (A)",
            min_value=-0.107,
            max_value=0.135,
            value=0.099,
            format="%.6f"
        )

        voltage = st.number_input(
            "Voltage (V)",
            min_value=2.497,
            max_value=4.205,
            value=3.700,
            format="%.6f"
        )

    st.write("Temperature reference: **23°C**")

    if st.button("Predict Next Reading", type="primary"):

        with st.spinner("Loading and training prediction models..."):

            trained_models = train_prediction_models()

        input_data = pd.DataFrame(
            [[duration, step, current, voltage]],
            columns=features
        )

        current_model = trained_models["Current"][model_choice]
        voltage_model = trained_models["Voltage"][model_choice]

        predicted_current = current_model.predict(input_data)[0]
        predicted_voltage = voltage_model.predict(input_data)[0]

        st.success("Prediction completed.")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Next Current",
            f"{predicted_current:.6f} A"
        )

        col2.metric(
            "Predicted Next Voltage",
            f"{predicted_voltage:.6f} V"
        )

        st.caption(
            "These are model estimates, not measured future readings."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "AI-Enabled Battery Prediction System"
)

st.sidebar.caption(
    "Recorded dataset prototype | 23°C reference"
)
