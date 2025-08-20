import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg.variants import frequency as dfg_freq_vis
from pm4py.visualization.dfg.variants import performance as dfg_perf_vis
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def simulate_parallel_faster(df, transitions):
    df = df.copy()
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True).dt.tz_localize(None)
    df["next_activity"] = df.groupby("case:concept:name")["concept:name"].shift(-1)
    df["next_index"] = df.index + 1

    parallel_map = pd.DataFrame()

    for (src, tgt) in transitions:
        mask = (df["concept:name"] == src) & (df["next_activity"] == tgt)
        temp = df.loc[mask, ["case:concept:name", "time:timestamp", "next_index"]].rename(
            columns={"time:timestamp": "src_time", "next_index": "target_idx"}
        )
        parallel_map = pd.concat([parallel_map, temp], axis=0)

    if not parallel_map.empty:
        parallel_map = parallel_map.drop_duplicates("target_idx")
        df.loc[parallel_map["target_idx"], "time:timestamp"] = pd.to_datetime(parallel_map["src_time"]).values

    return df

def run_process_mining_pipeline(file_path, case_col, activity_col, timestamp_col):
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
    df = df.rename(columns={case_col: "case:concept:name", activity_col: "concept:name", timestamp_col: "time:timestamp"})
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors='coerce', utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["time:timestamp"])
    df = dataframe_utils.convert_timestamp_columns_in_df(df)

    event_log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
    print(f"\n✅ Loaded event log: {len(event_log)} cases, {len(df)} events")

    net, im, fm = alpha_miner.apply(event_log)
    pn_visualizer.view(pn_visualizer.apply(net, im, fm))

    replay_result = token_replay.apply(event_log, net, im, fm)
    fitness_scores = [res['trace_fitness'] for res in replay_result if 'trace_fitness' in res]
    print(f"\n🎯 Fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
    print(f"📏 Precision: {precision_evaluator.apply(event_log, net, im, fm):.4f}")

    heu_net = heuristics_miner.apply_heu(event_log)
    hn_visualizer.view(hn_visualizer.apply(heu_net))

    dfg = dfg_discovery.apply(event_log)
    dfg_visualization.view(dfg_visualization.apply(dfg, log=event_log, variant=dfg_freq_vis))

    perf_dfg = dfg_discovery.apply(df, variant=dfg_discovery.Variants.PERFORMANCE)
    dfg_visualization.view(dfg_visualization.apply(perf_dfg, log=event_log, variant=dfg_perf_vis))

    bottlenecks = {k: v for k, v in perf_dfg.items() if v > 6 * 60 * 60}
    print("\n🐢 Detected bottlenecks (>6 hrs):")
    for (src, tgt), duration in bottlenecks.items():
        print(f"{src} → {tgt}: {duration/3600:.2f} hrs")

    df_fixed = df.copy()
    df_fixed["time:timestamp"] = pd.to_datetime(df_fixed["time:timestamp"], utc=True).dt.tz_localize(None)
    df_fixed["next_activity"] = df_fixed.groupby("case:concept:name")["concept:name"].shift(-1)
    for (src, tgt) in bottlenecks:
        mask = (df_fixed["concept:name"] == src) & (df_fixed["next_activity"] == tgt)
        affected_idx = df_fixed[mask].index
        next_idx = affected_idx + 1
        next_idx = next_idx[next_idx < len(df_fixed)]
        df_fixed.loc[next_idx, "time:timestamp"] = pd.to_datetime(df_fixed.loc[affected_idx, "time:timestamp"]).values + pd.Timedelta(minutes=30)

    df_optimized = simulate_parallel_faster(df_fixed, bottlenecks)
    df_optimized = dataframe_utils.convert_timestamp_columns_in_df(df_optimized)

    event_log_optimized = log_converter.apply(df_optimized, variant=log_converter.Variants.TO_EVENT_LOG)
    final_perf_dfg = dfg_discovery.apply(df_optimized, variant=dfg_discovery.Variants.PERFORMANCE)
    dfg_visualization.view(dfg_visualization.apply(final_perf_dfg, log=event_log_optimized, variant=dfg_perf_vis))

    df_optimized.to_csv("final_optimized_event_log.csv", index=False)
    print("\n📁 Saved optimized log: 'final_optimized_event_log.csv'")

    G = nx.DiGraph()
    G.add_edges_from(final_perf_dfg.keys())
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue",
            font_size=10, font_weight="bold", arrows=True, edge_color="gray")
    plt.title("🚀 Optimized Process Model (Fixed & Parallelized)")
    plt.tight_layout()
    plt.show()

    return df_optimized

def add_ml_prediction(df_optimized):
    print("\n🔍 Starting ML: Outcome Prediction")
    case_activities = df_optimized.groupby("case:concept:name")["concept:name"].apply(list)
    labels = case_activities.apply(lambda x: "Approved" if "Payment Sent" in x else "Rejected")
    df_optimized["label"] = df_optimized["case:concept:name"].map(labels)

    features = df_optimized.groupby("case:concept:name").agg({
        "concept:name": ["count", "nunique"],
        "time:timestamp": lambda x: x.max() - x.min()
    })
    features.columns = ["num_steps", "unique_activities", "duration"]
    features["duration_mins"] = features["duration"].dt.total_seconds() / 60
    features = features.drop(columns="duration")
    features["label"] = labels
    features = features.dropna()

    X = features[["num_steps", "unique_activities", "duration_mins"]]
    y = features["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if X_scaled.shape[0] != len(y) or len(y) == 0:
        print("❌ Mismatch or empty feature set. Skipping ML.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n🧠 Model Performance:")
    print(classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, scaler

def predict_case_duration(df_optimized):
    print("\n⏱ ML: Predicting Case Duration (Regression)")
    case_activities = df_optimized.groupby("case:concept:name")["concept:name"].apply(list)
    labels = case_activities.apply(lambda x: "Approved" if "Payment Sent" in x else "Rejected")
    df_optimized["label"] = df_optimized["case:concept:name"].map(labels)

    features = df_optimized.groupby("case:concept:name").agg({
        "concept:name": ["count", "nunique"],
        "time:timestamp": lambda x: x.max() - x.min()
    })
    features.columns = ["num_steps", "unique_activities", "duration"]
    features["duration_mins"] = features["duration"].dt.total_seconds() / 60
    features = features.drop(columns="duration").dropna()

    X = features[["num_steps", "unique_activities"]]
    y = features["duration_mins"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("X_scaled shape:", X_scaled.shape)
    print("y length:", len(y))

    if X_scaled.shape[0] != len(y) or len(y) == 0:
        print("❌ Mismatch or empty feature set. Skipping ML.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train)

    y_pred = reg_model.predict(X_test)
    print("\n📊 Regression Evaluation:")
    print(f"📉 Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} mins")
    print(f"📈 R² Score: {r2_score(y_test, y_pred):.4f}")

    return reg_model, scaler

if __name__ == '__main__':
    df_opt = run_process_mining_pipeline(
        r"C:\Users\KIRTI\Desktop\pbl\Insurance_claims_event_log.xlsx",
        case_col="case_id",
        activity_col="activity_name",
        timestamp_col="timestamp"
    )
    add_ml_prediction(df_opt)
    predict_case_duration(df_opt)
