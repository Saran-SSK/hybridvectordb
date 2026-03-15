# HybridVectorDB Frontend Dashboard

## Overview

This is the complete HTML dashboard for HybridVectorDB, integrated without any changes to preserve the original design and functionality.

## Features

- **Complete Dashboard**: All pages and functionality preserved exactly as designed
- **Real-time Monitoring**: Live performance metrics and system health
- **Vector Management**: Upload, search, and manage vector collections
- **Infrastructure Monitoring**: GPU, CPU, and cluster monitoring
- **Configuration Management**: System settings and API key management
- **Search Playground**: Interactive vector similarity search
- **Alerts System**: Real-time system notifications

## Quick Start

### Option 1: Python Server (Recommended)
```bash
cd frontend
python serve.py
```

### Option 2: Node.js Server
```bash
cd frontend
npm install
npm start
```

### Option 3: Simple HTTP Server
```bash
cd frontend
python -m http.server 3000
```

## Access Points

- **Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **Health Check**: http://localhost:8080/health

## Architecture

```
Frontend (http://localhost:3000)
    ↓ HTTP Requests
Backend API (http://localhost:8080)
    ↓
HybridVectorDB Core
```

## Integration Details

### Frontend Structure
- **index.html**: Complete dashboard (1,840 lines)
- **serve.py**: Python HTTP server with API proxy
- **package.json**: Node.js server configuration

### API Integration
The frontend is configured to communicate with the HybridVectorDB backend API:

- **Health Monitoring**: `/api/health`
- **Performance Metrics**: `/api/metrics`
- **Vector Operations**: `/api/vectors`, `/api/search`
- **Configuration**: `/api/config`
- **Statistics**: `/api/stats`

### Features Preserved

#### Dashboard Pages
- **Overview**: System metrics and performance charts
- **Performance**: Latency and throughput analysis
- **Alerts**: System notifications and warnings
- **Vectors**: Collection management and upload
- **Search**: Interactive search playground
- **Indexes**: Index management and configuration
- **Router**: Query routing visualization
- **GPU Monitor**: GPU utilization and memory
- **Cluster**: Cluster status and nodes
- **Configuration**: System settings management
- **API Keys**: Authentication management

#### Visual Components
- **Real-time Charts**: Performance metrics visualization
- **Metric Cards**: Key performance indicators
- **Alert System**: Live notifications
- **Upload Interface**: Batch vector upload
- **Search Interface**: Vector similarity search
- **Configuration Panels**: System settings

## Backend Requirements

The frontend expects the following API endpoints to be available:

```python
# Health check
GET /api/health

# Performance metrics
GET /api/metrics

# Vector operations
POST /api/vectors     # Add vectors
POST /api/search      # Search vectors
GET /api/vectors/count # Get vector count

# Index management
GET /api/index/info   # Get index information
POST /api/index/save  # Save index
POST /api/index/load  # Load index

# Configuration
GET /api/config       # Get configuration
PUT /api/config       # Update configuration

# Statistics
GET /api/stats         # Get statistics
GET /api/stats/performance  # Performance stats
GET /api/stats/memory       # Memory stats

# Benchmarking
POST /api/benchmark    # Run benchmark
GET /api/benchmark/results  # Get benchmark results
```

## Development

### Frontend Technologies
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with CSS variables
- **JavaScript**: Vanilla JS with Chart.js
- **Chart.js**: Data visualization
- **Google Fonts**: Typography (JetBrains Mono, Syne)

### Design Features
- **Dark Theme**: Professional dark UI design
- **Responsive**: Mobile-friendly layout
- **Real-time Updates**: Live data simulation
- **Interactive Charts**: Performance visualization
- **Modular Components**: Reusable UI elements

## Customization

The frontend can be customized by modifying the `index.html` file while preserving the core structure and functionality.

### API Configuration
Update the API endpoints in the JavaScript section of `index.html` to match your backend configuration.

### Styling
Modify the CSS variables in the `<style>` section to customize colors and themes.

### Data Sources
Update the JavaScript functions to connect to real data sources instead of simulated data.

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the backend server has CORS enabled
2. **API Connection**: Verify backend is running on port 8080
3. **Static Files**: Check file permissions and paths
4. **Browser Cache**: Clear cache if dashboard doesn't update

### Debug Mode
Open browser developer tools to view:
- Network requests
- Console errors
- API responses

## Production Deployment

For production deployment:

1. **Static Hosting**: Deploy to CDN or static hosting service
2. **Domain Configuration**: Update API URLs for production
3. **HTTPS**: Enable SSL/TLS for secure connections
4. **Caching**: Configure browser caching policies
5. **Performance**: Optimize assets and enable compression

## License

MIT License - see main project LICENSE file for details.
