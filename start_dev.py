#!/usr/bin/env python3
"""
Development launcher for HybridVectorDB Frontend + Backend
"""

import subprocess
import sys
import time
import os
import webbrowser
from threading import Thread

def start_backend():
    """Start the backend API server"""
    print("Starting HybridVectorDB Backend API...")
    try:
        subprocess.run([sys.executable, "simple_server.py"], check=True)
    except KeyboardInterrupt:
        print("\nBackend server stopped.")
    except Exception as e:
        print(f"Backend error: {e}")

def start_frontend():
    """Start the frontend development server"""
    print("Starting HybridVectorDB Frontend...")
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    
    try:
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Check if node_modules exists
        if not os.path.exists("node_modules"):
            print("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start frontend dev server
        subprocess.run(["npm", "run", "dev"], check=True)
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")
    except Exception as e:
        print(f"Frontend error: {e}")
    finally:
        # Change back to original directory
        os.chdir("..")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("HybridVectorDB Development Launcher")
    print("=" * 60)
    print()
    
    print("This will start:")
    print("1. Backend API server on http://localhost:8080")
    print("2. Frontend dashboard on http://localhost:3000")
    print()
    
    # Check dependencies
    try:
        import flask
        import flask_cors
        print("✓ Backend dependencies found")
    except ImportError:
        print("✗ Installing backend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors"], check=True)
    
    # Check if frontend directory exists
    if not os.path.exists("frontend"):
        print("✗ Frontend directory not found!")
        return
    
    print("\nStarting servers...")
    print("Press Ctrl+C to stop both servers")
    print()
    
    # Start backend in a separate thread
    backend_thread = Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(2)
    
    # Start frontend in main thread (this will block)
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
