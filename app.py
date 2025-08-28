from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
import io
import base64
import json
import os
from datetime import datetime

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score
)

app = Flask(__name__)

# Global variables to store analysis results
analysis_results = {}

def simulate_parallel_faster(df, transitions):
    """Simulate parallel execution for faster processing"""
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

def run_process_mining_analysis(file_path, case_col, activity_col, timestamp_col):
    """Run the complete process mining analysis"""
    global analysis_results
    
    try:
        # Load data
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Rename columns to standard format
        df = df.rename(columns={
            case_col: "case:concept:name", 
            activity_col: "concept:name", 
            timestamp_col: "time:timestamp"
        })
        
        # Clean data
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors='coerce', utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["time:timestamp"])
        df = dataframe_utils.convert_timestamp_columns_in_df(df)

        # Convert to event log
        event_log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
        
        # Alpha miner
        net, im, fm = alpha_miner.apply(event_log)
        
        # Fitness and precision
        replay_result = token_replay.apply(event_log, net, im, fm)
        fitness_scores = [res['trace_fitness'] for res in replay_result if 'trace_fitness' in res]
        fitness = sum(fitness_scores)/len(fitness_scores) if fitness_scores else 0
        precision = precision_evaluator.apply(event_log, net, im, fm)
        
        # Heuristics miner
        heu_net = heuristics_miner.apply_heu(event_log)
        
        # DFG discovery
        dfg = dfg_discovery.apply(event_log)
        perf_dfg = dfg_discovery.apply(df, variant=dfg_discovery.Variants.PERFORMANCE)
        
        # Detect bottlenecks
        bottlenecks = {k: v for k, v in perf_dfg.items() if v > 6 * 60 * 60}  # >6 hours
        
        # Optimize process
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
        
        # Save optimized log
        df_optimized.to_csv("final_optimized_event_log.csv", index=False)
        
        # ML Analysis
        ml_results = perform_ml_analysis(df_optimized)
        
        # Store results
        analysis_results = {
            'original_data': df,
            'optimized_data': df_optimized,
            'event_log': event_log,
            'fitness': fitness,
            'precision': precision,
            'bottlenecks': bottlenecks,
            'dfg': dfg,
            'perf_dfg': perf_dfg,
            'ml_results': ml_results,
            'cases_count': len(event_log),
            'events_count': len(df),
            'activities': df['concept:name'].unique().tolist()
        }
        
        return True, "Analysis completed successfully"
        
    except Exception as e:
        return False, str(e)

def perform_ml_analysis(df_optimized):
    """Perform ML analysis for outcome prediction and duration prediction"""
    try:
        # Outcome prediction
        case_activities = df_optimized.groupby("case:concept:name")["concept:name"].apply(list)
        labels = case_activities.apply(lambda x: "Approved" if "Payment Sent" in x else "Rejected")
        
        features = df_optimized.groupby("case:concept:name").agg({
            "concept:name": ["count", "nunique"],
            "time:timestamp": lambda x: x.max() - x.min()
        })
        features.columns = ["num_steps", "unique_activities", "duration"]
        features["duration_mins"] = features["duration"].dt.total_seconds() / 60
        features = features.drop(columns="duration")
        features["label"] = labels
        features = features.dropna()

        ml_results = {}
        
        if len(features) > 0:
            X = features[["num_steps", "unique_activities", "duration_mins"]]
            y = features["label"]
            
            if len(X) > 4:  # Need minimum samples for train/test split
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                ml_results['classification'] = {
                    'accuracy': (y_pred == y_test).mean(),
                    'prediction_distribution': pd.Series(y_pred).value_counts().to_dict()
                }
            
            # Duration prediction
            X_dur = features[["num_steps", "unique_activities"]]
            y_dur = features["duration_mins"]
            
            if len(X_dur) > 4:
                scaler_dur = StandardScaler()
                X_dur_scaled = scaler_dur.fit_transform(X_dur)
                
                X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(X_dur_scaled, y_dur, test_size=0.25, random_state=42)
                reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
                reg_model.fit(X_train_dur, y_train_dur)
                
                y_pred_dur = reg_model.predict(X_test_dur)
                
                ml_results['regression'] = {
                    'mae': mean_absolute_error(y_test_dur, y_pred_dur),
                    'r2': r2_score(y_test_dur, y_pred_dur)
                }
        
        return ml_results
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html', results=analysis_results)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        file = request.files['file']
        case_col = request.form['case_col']
        activity_col = request.form['activity_col']
        timestamp_col = request.form['timestamp_col']
        
        if file:
            filename = file.filename
            file.save(filename)
            
            success, message = run_process_mining_analysis(filename, case_col, activity_col, timestamp_col)
            
            if success:
                return jsonify({'status': 'success', 'message': message})
            else:
                return jsonify({'status': 'error', 'message': message})
        
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/analyze_existing')
def analyze_existing():
    """Analyze the existing Insurance claims file"""
    try:
        success, message = run_process_mining_analysis(
            "Insurance_claims_event_log.xlsx",
            "case_id",
            "activity_name", 
            "timestamp"
        )
        
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_statistics')
def get_statistics():
    """Get analysis statistics"""
    if not analysis_results:
        return jsonify({'error': 'No analysis results available'})
    
    stats = {
        'cases_count': analysis_results.get('cases_count', 0),
        'events_count': analysis_results.get('events_count', 0),
        'fitness': round(analysis_results.get('fitness', 0), 4),
        'precision': round(analysis_results.get('precision', 0), 4),
        'bottlenecks_count': len(analysis_results.get('bottlenecks', {})),
        'activities': analysis_results.get('activities', [])
    }
    
    return jsonify(stats)

@app.route('/get_bottlenecks')
def get_bottlenecks():
    """Get bottleneck information"""
    if not analysis_results or 'bottlenecks' not in analysis_results:
        return jsonify({'error': 'No bottleneck data available'})
    
    bottlenecks = []
    for (src, tgt), duration in analysis_results['bottlenecks'].items():
        bottlenecks.append({
            'source': src,
            'target': tgt,
            'duration_hours': round(duration / 3600, 2)
        })
    
    return jsonify(bottlenecks)

@app.route('/get_process_graph')
def get_process_graph():
    """Generate process flow graph"""
    if not analysis_results or 'perf_dfg' not in analysis_results:
        return jsonify({'error': 'No process data available'})
    
    try:
        perf_dfg = analysis_results['perf_dfg']
        
        # Check if perf_dfg is empty
        if not perf_dfg:
            return jsonify({'error': 'No process flow data available'})
        
        # Create nodes and edges for plotly
        nodes = set()
        edges = []
        
        for (src, tgt), duration in perf_dfg.items():
            nodes.add(src)
            nodes.add(tgt)
            edges.append({
                'source': src,
                'target': tgt,
                'duration': round(duration / 3600, 2) if duration > 0 else 0
            })
        
        if not nodes:
            return jsonify({'error': 'No nodes found in process data'})
        
        nodes_list = list(nodes)
        
        # Create plotly network graph
        G = nx.DiGraph()
        G.add_edges_from([(edge['source'], edge['target']) for edge in edges])
        
        if len(G.nodes()) == 0:
            return jsonify({'error': 'Empty graph generated'})
        
        # Use circular layout if spring layout fails
        try:
            pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        except:
            pos = nx.circular_layout(G)
        
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Find duration for this edge
                duration = next((e['duration'] for e in edges if e['source'] == edge[0] and e['target'] == edge[1]), 0)
                edge_info.append(f"{edge[0]} → {edge[1]}: {duration}h")

        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node[:20] + "..." if len(node) > 20 else node)  # Truncate long names
                node_info.append(f"Activity: {node}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Process Flow'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='black')
            ),
            name='Activities'
        )

        layout = go.Layout(
            title=dict(
                text='Process Flow Graph',
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Process Mining Analysis - Hover over nodes for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                font=dict(size=12, color="gray")
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        
        return jsonify(fig.to_dict())
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_process_graph: {error_details}")
        return jsonify({'error': f'Graph generation failed: {str(e)}'})

@app.route('/debug_analysis')
def debug_analysis():
    """Debug endpoint to check analysis results"""
    debug_info = {
        'analysis_results_exists': bool(analysis_results),
        'analysis_keys': list(analysis_results.keys()) if analysis_results else [],
        'perf_dfg_exists': 'perf_dfg' in analysis_results if analysis_results else False,
        'perf_dfg_length': len(analysis_results.get('perf_dfg', {})) if analysis_results else 0,
        'sample_perf_dfg': dict(list(analysis_results.get('perf_dfg', {}).items())[:3]) if analysis_results else {}
    }
    return jsonify(debug_info)

@app.route('/get_activity_details')
def get_activity_details():
    """Get detailed activity information"""
    if not analysis_results or 'original_data' not in analysis_results:
        return jsonify({'error': 'No analysis data available'})
    
    try:
        df = analysis_results['original_data']
        
        # Calculate activity frequencies
        activity_counts = df['concept:name'].value_counts().to_dict()
        
        # Calculate average duration per activity (if possible)
        activity_durations = {}
        for activity in df['concept:name'].unique():
            activity_data = df[df['concept:name'] == activity]
            if len(activity_data) > 1:
                # Calculate average time between this activity and next
                activity_durations[activity] = 'Variable'
            else:
                activity_durations[activity] = 'Single occurrence'
        
        # Get activity sequence
        activity_sequence = df.groupby('case:concept:name')['concept:name'].apply(list).iloc[0] if len(df) > 0 else []
        
        details = {
            'activity_counts': activity_counts,
            'activity_durations': activity_durations,
            'total_unique_activities': len(df['concept:name'].unique()),
            'most_frequent_activity': df['concept:name'].value_counts().index[0] if len(df) > 0 else None,
            'activity_sequence_sample': activity_sequence
        }
        
        return jsonify(details)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_optimized')
def download_optimized():
    """Download optimized event log"""
    if os.path.exists("final_optimized_event_log.csv"):
        return send_file("final_optimized_event_log.csv", as_attachment=True)
    else:
        return jsonify({'error': 'Optimized file not found'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
