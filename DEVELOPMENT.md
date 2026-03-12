# HybridVectorDB Development Guide

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm/yarn/pnpm

### Option 1: Automatic Launch (Recommended)

```bash
# From the root directory
python start_dev.py
```

This will automatically:
- Install backend dependencies (Flask, Flask-CORS)
- Install frontend dependencies (npm packages)
- Start backend API server on http://localhost:8080
- Start frontend dashboard on http://localhost:3000

### Option 2: Manual Launch

#### Backend Server

```bash
# Install dependencies
pip install flask flask-cors

# Start backend
python simple_server.py
```

Backend will be available at:
- API: http://localhost:8080
- Health: http://localhost:8080/health
- Metrics: http://localhost:8080/metrics

#### Frontend Dashboard

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at:
- Dashboard: http://localhost:3000
- Network: http://192.168.1.11:3000

## Development Features

### Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Performance metrics |
| `/stats` | GET | Database statistics |
| `/vectors` | POST | Add vectors |
| `/search` | POST | Search vectors |
| `/vectors/count` | GET | Vector count |
| `/index/info` | GET | Index information |
| `/benchmark` | POST | Run benchmark |
| `/config` | GET/PUT | Configuration |

### Frontend Dashboard

- **Real-time Monitoring**: Live performance metrics and charts
- **Vector Management**: Add, search, and manage vectors
- **Benchmarking**: Run and analyze performance benchmarks
- **Configuration**: Adjust database settings
- **Alerts Panel**: System notifications and warnings
- **Routing Insights**: CPU/GPU query distribution

### API Examples

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Get Metrics
```bash
curl http://localhost:8080/metrics
```

#### Search Vectors
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, ...],
    "k": 10
  }'
```

#### Add Vectors
```bash
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    "metadata": [{"source": "test"}, {"source": "test"}]
  }'
```

## Project Structure

```
hybridvectordb/
├── frontend/                 # Next.js dashboard
│   ├── src/
│   │   ├── app/             # App router pages
│   │   ├── components/      # React components
│   │   └── lib/            # Utilities and API client
│   ├── package.json
│   └── README.md
├── src/cpp/                 # C++ implementation
├── hybridvectordb/          # Python package
├── simple_server.py         # Development API server
├── start_dev.py            # Development launcher
└── DEVELOPMENT.md          # This file
```

## Development Workflow

### 1. Make Changes
- Edit backend code in `simple_server.py` or Python package
- Edit frontend code in `frontend/src/`

### 2. Test Changes
- Backend changes are reflected immediately (auto-restart)
- Frontend changes trigger hot reload in browser

### 3. API Integration
- Frontend automatically connects to `http://localhost:8080`
- CORS is enabled for development
- Mock data available when backend is down

### 4. Production Build
```bash
# Frontend
cd frontend
npm run build

# Backend (use production server)
# See docs/production.md for production deployment
```

## Troubleshooting

### Backend Issues

**Port 8080 already in use:**
```bash
# Find process using port 8080
netstat -ano | findstr :8080

# Kill the process
taskkill /PID <PID> /F
```

**Missing dependencies:**
```bash
pip install flask flask-cors
```

### Frontend Issues

**Port 3000 already in use:**
```bash
# Kill Node.js processes
taskkill /F /IM node.exe

# Or use different port
npm run dev -- -p 3001
```

**Dependencies not installed:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Build errors:**
```bash
# Clear Next.js cache
cd frontend
rm -rf .next
npm run dev
```

### Connection Issues

**CORS errors:**
- Backend server includes CORS headers
- Make sure backend is running on port 8080

**API not responding:**
- Check backend console for errors
- Verify endpoints are accessible via curl
- Check network tab in browser dev tools

## Environment Variables

### Backend (.env)
```env
FLASK_ENV=development
FLASK_DEBUG=1
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080
NEXT_PUBLIC_APP_NAME=HybridVectorDB Console
NEXT_PUBLIC_APP_VERSION=0.5.0
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test with development servers
5. Submit pull request

## Next Steps

- [ ] Connect to real HybridVectorDB backend
- [ ] Add authentication/authorization
- [ ] Implement real-time WebSocket updates
- [ ] Add more comprehensive benchmarking
- [ ] Deploy to production environment

## Support

For issues and questions:
- Check this development guide
- Review the main README.md
- Open an issue on GitHub
