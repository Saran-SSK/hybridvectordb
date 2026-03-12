# HybridVectorDB Frontend

A modern Next.js dashboard for monitoring and managing HybridVectorDB.

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env.local
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

4. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Features

- 🚀 Real-time performance monitoring
- 📊 Interactive charts and visualizations
- 🔧 Configuration management
- 📈 Benchmark results and analytics
- 🎯 Vector search and management
- 🔄 Hybrid routing insights
- 📱 Responsive design

## Backend Connection

The frontend automatically connects to the HybridVectorDB backend running on `http://localhost:8080`. Make sure the backend is running before starting the frontend.

### Start Backend

```bash
# From the root directory
cd "c:/Users/Admin/Desktop/vector database"
python -m hybridvectordb
```

### Start Frontend

```bash
# From the frontend directory
cd frontend
npm run dev
```

## Environment Variables

Create a `.env.local` file with the following variables:

```env
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080
NEXT_PUBLIC_APP_NAME=HybridVectorDB Console
NEXT_PUBLIC_APP_VERSION=0.5.0
```

## Build for Production

```bash
npm run build
npm start
```

## Technology Stack

- **Framework:** Next.js 16 with App Router
- **Styling:** Tailwind CSS
- **UI Components:** Radix UI
- **Charts:** Recharts
- **Icons:** Lucide React
- **HTTP Client:** Axios
- **TypeScript:** Full type safety
