import os
import time
import queue
import threading
import numpy as np
from collections import deque

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import sounddevice as sd
import webrtcvad
import torch
import librosa
import plotly.graph_objs as go
from source.components.inference import InferenceThread

# ─── Global state and queues ──────────────────────────────────────────────────
RUNNING = False
THREAD = None
RESULTS_QUEUE = queue.Queue()  # For passing results from inference thread to UI
AUDIO_BUFFER = deque(maxlen=4000)  # About 0.25 seconds @16kHz
AUDIO_QUEUE = queue.Queue()  # For passing audio data from inference thread to UI


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Keyword Spotting Demo", style={'textAlign': 'center'}),
    
    html.Div([
        html.Button("Start Listening", id="start-button", n_clicks=0, 
                   style={'fontSize': '18px', 'margin': '10px', 'padding': '10px 20px'}),
        html.Button("Stop", id="stop-button", n_clicks=0, disabled=True,
                   style={'fontSize': '18px', 'margin': '10px', 'padding': '10px 20px'}),
    ], style={'textAlign': 'center', 'margin': '20px'}),
    
    html.Div([
        html.Div(id="recording-indicator", 
                 style={'width': '20px', 'height': '20px', 'borderRadius': '50%', 
                        'backgroundColor': '#ff4d4f', 'display': 'inline-block'}),
        html.Span(" Recording Status", style={'marginLeft': '10px', 'verticalAlign': 'middle'})
    ], style={'margin': '20px', 'display': 'flex', 'alignItems': 'center'}),
    
    html.Div([
        html.H3("Status:"),
        html.Div(id="status-display", children="Not listening", 
                style={'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#f0f0f0'})
    ], style={'margin': '20px'}),
    
    html.Div([
        html.H3("Live Audio:"),
        dcc.Graph(
            id='waveform-graph',
            figure={
                'data': [{'y': [0]*1000, 'type': 'line', 'name': 'Waveform'}],
                'layout': {
                    'title': 'Live Audio Waveform',
                    'height': 200,
                    'margin': {'l': 30, 'r': 30, 't': 30, 'b': 30},
                    'yaxis': {'range': [-1, 1]},
                    'xaxis': {'showticklabels': False}
                }
            },
            config={'displayModeBar': False}
        )
    ], style={'margin': '20px'}),
    
    html.Div([
        html.H3("Detected Keywords:"),
        html.Div(id="results-display", style={'maxHeight': '400px', 'overflowY': 'auto'})
    ], style={'margin': '20px'}),
    
    # Hidden div for storing results
    html.Div(id="results-store", style={'display': 'none'}),
    
    # Interval for updating UI
    dcc.Interval(id="update-interval", interval=100, n_intervals=0)  # Update more frequently
])

# ─── Callbacks ──────────────────────────────────────────────────────────
@app.callback(
    [Output("start-button", "disabled"),
     Output("stop-button", "disabled"),
     Output("status-display", "children"),
     Output("status-display", "style")],
    [Input("start-button", "n_clicks"),
     Input("stop-button", "n_clicks")],
    [State("start-button", "disabled")]
)
def toggle_listening(start_clicks, stop_clicks, start_disabled):
    global RUNNING, THREAD, AUDIO_BUFFER
    
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        # No button clicked yet
        return False, True, "Not listening", {'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "start-button" and not start_disabled:
        # Start button clicked
        AUDIO_BUFFER.clear()  # Clear audio buffer
        THREAD = InferenceThread(
            model_path='/home/clinton-mwangi/Desktop/msc-telecom/1.2/Speech/kws/keyword-spotting/source/components/model2.pth',
            results_queue=RESULTS_QUEUE,
            audio_queue=AUDIO_QUEUE
        )
        THREAD.start()
        RUNNING = True
        return True, False, "Listening for keywords...", {'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#d4f7d4'}
    
    elif button_id == "stop-button":
        # Stop button clicked
        if THREAD and THREAD.is_alive():
            THREAD.stop()
            THREAD.join(timeout=1.0)
        RUNNING = False
        return False, True, "Stopped listening", {'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#f7d4d4'}
    
    # Default return
    return False, True, "Not listening", {'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}

@app.callback(
    Output("recording-indicator", "style"),
    Input("update-interval", "n_intervals"),
    State("status-display", "children")
)
def update_recording_indicator(n_intervals, status):
    base_style = {'width': '20px', 'height': '20px', 'borderRadius': '50%', 'display': 'inline-block'}
    
    if "Listening" in status:
        # Check if we have recent speech detection
        if len(AUDIO_BUFFER) > 100:
            chunk = list(AUDIO_BUFFER)[-100:]
            if np.abs(chunk).mean() > 0.05:  # Adjust threshold as needed
                # Active speech detected
                return {**base_style, 'backgroundColor': '#52c41a', 'boxShadow': '0 0 10px #52c41a'}
        
        # Listening but no active speech
        return {**base_style, 'backgroundColor': '#faad14'}
    else:
        # Not listening
        return {**base_style, 'backgroundColor': '#ff4d4f'}

@app.callback(
    Output('waveform-graph', 'figure'),
    Input('update-interval', 'n_intervals')
)
def update_waveform(n_intervals):
    global AUDIO_BUFFER
    
    # Get any new audio frames
    while not AUDIO_QUEUE.empty():
        try:
            frame = AUDIO_QUEUE.get_nowait()
            AUDIO_BUFFER.extend(frame)
        except queue.Empty:
            break
    
    # Create the waveform figure
    return {
        'data': [
            {
                'y': list(AUDIO_BUFFER),
                'type': 'line',
                'name': 'Waveform',
                'line': {'color': '#2ca02c'}
            }
        ],
        'layout': {
            'title': 'Live Audio Waveform',
            'height': 200,
            'margin': {'l': 30, 'r': 30, 't': 30, 'b': 30},
            'yaxis': {'range': [-1, 1]},
            'xaxis': {'showticklabels': False},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(240,240,240,0.8)'
        }
    }

@app.callback(
    Output("results-display", "children"),
    Input("update-interval", "n_intervals"),
    State("results-display", "children")
)
def update_results(n_intervals, current_results):
    if current_results is None:
        current_results = []
    
    # Get any new results from the queue
    new_results = []
    while not RESULTS_QUEUE.empty():
        result = RESULTS_QUEUE.get()
        
        # Different styling for speech detection vs keyword detection
        if result.get('index', 0) == -1:
            # Speech detection notification
            style = {'padding': '5px', 'margin': '5px', 'backgroundColor': '#f7f7f7', 'borderRadius': '5px'}
        else:
            # Get confidence if available
            confidence = result.get('confidence', 0)
            
            # Color gradient from yellow (low confidence) to green (high confidence)
            if confidence >= 0.7:
                color = '#d4f7d4'  # Green for high confidence
            elif confidence >= 0.5:
                color = '#e6f7ff'  # Blue for medium confidence
            else:
                color = '#fff7e6'  # Yellow for lower confidence
                
            style = {
                'padding': '10px', 
                'margin': '5px', 
                'backgroundColor': color, 
                'borderRadius': '5px',
                'fontWeight': 'bold'
            }
        
        new_results.append(
            html.Div([
                html.Span(f"{result['timestamp']} - ", style={'fontWeight': 'bold'}),
                html.Span(f"{result['label']}")
            ], style=style)
        )
    
    # Add new results to the top
    return new_results + current_results if current_results else new_results

# ─── Run the app ────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
