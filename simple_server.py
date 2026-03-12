#!/usr/bin/env python3
"""
Simple Flask server for HybridVectorDB API demonstration
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import time
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Mock database state
vectors_db = []
vector_count = 1247832

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'uptime': '14d 7h 23m',
        'version': '0.5.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics"""
    return jsonify({
        'totalVectors': vector_count,
        'throughput': 2340 + random.randint(-100, 100),
        'avgLatency': round(1.3 + random.uniform(-0.2, 0.2), 2),
        'memoryUsage': round(12.4 + random.uniform(-1, 1), 1),
        'gpuUsage': 67 + random.randint(-10, 10),
        'cpuUsage': 42 + random.randint(-5, 5),
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    return jsonify({
        'vector_count': vector_count,
        'index_type': 'ivf',
        'dimension': 768,
        'distance_metric': 'cosine',
        'gpu_enabled': True,
        'memory_usage': '12.4 GB',
        'last_updated': datetime.utcnow().isoformat()
    })

@app.route('/vectors', methods=['POST'])
def add_vectors():
    """Add vectors to database"""
    data = request.get_json()
    vectors = data.get('vectors', [])
    
    global vector_count
    vector_count += len(vectors)
    
    return jsonify({
        'added': len(vectors),
        'total_count': vector_count,
        'message': f'Successfully added {len(vectors)} vectors'
    })

@app.route('/search', methods=['POST'])
def search_vectors():
    """Search vectors"""
    data = request.get_json()
    query = data.get('query', [])
    k = data.get('k', 10)
    
    # Generate mock search results
    results = []
    for i in range(k):
        results.append({
            'id': f'vec_{i:06d}',
            'distance': round(random.uniform(0.01, 0.5), 3),
            'score': round(random.uniform(0.5, 0.99), 3),
            'metadata': {
                'source': 'mock_data',
                'category': 'test'
            }
        })
    
    return jsonify({
        'results': results,
        'query_time_ms': round(random.uniform(1, 5), 2),
        'total_results': len(results)
    })

@app.route('/vectors/count', methods=['GET'])
def get_vector_count():
    """Get vector count"""
    return jsonify({
        'count': vector_count,
        'last_updated': datetime.utcnow().isoformat()
    })

@app.route('/index/info', methods=['GET'])
def get_index_info():
    """Get index information"""
    return jsonify({
        'type': 'ivf',
        'dimension': 768,
        'nlist': 1000,
        'nprobe': 10,
        'metric': 'cosine',
        'is_trained': True,
        'vector_count': vector_count
    })

@app.route('/benchmark', methods=['POST'])
def run_benchmark():
    """Run benchmark"""
    config = request.get_json()
    
    # Simulate benchmark running
    time.sleep(2)
    
    return jsonify({
        'benchmark_id': f'bench_{int(time.time())}',
        'status': 'completed',
        'results': {
            'dataset_size': config.get('num_vectors', 10000),
            'avg_latency_ms': round(random.uniform(1, 10), 2),
            'throughput_ops_per_sec': random.randint(1000, 5000),
            'recall': round(random.uniform(0.95, 0.99), 3),
            'memory_usage_mb': random.randint(100, 1000)
        },
        'completed_at': datetime.utcnow().isoformat()
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get configuration"""
    return jsonify({
        'dimension': 768,
        'index_type': 'ivf',
        'distance_metric': 'cosine',
        'use_gpu': True,
        'nlist': 1000,
        'nprobe': 10,
        'batch_size': 32
    })

@app.route('/config', methods=['PUT'])
def update_config():
    """Update configuration"""
    config = request.get_json()
    
    return jsonify({
        'message': 'Configuration updated successfully',
        'config': config,
        'updated_at': datetime.utcnow().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting HybridVectorDB API Server...")
    print("Frontend: http://localhost:3000")
    print("API: http://localhost:8080")
    print("Health: http://localhost:8080/health")
    print("Metrics: http://localhost:8080/metrics")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
