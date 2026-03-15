#!/usr/bin/env python3
"""
Simple HTTP server for HybridVectorDB HTML frontend
Serves the static HTML dashboard and proxies API requests to the backend
"""

import http.server
import socketserver
import urllib.parse
import urllib.request
import json
import os
from pathlib import Path

class HybridVectorDBHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    def do_GET(self):
        if self.path.startswith('/api/'):
            self.proxy_request()
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        if self.path.startswith('/api/'):
            self.proxy_request()
        else:
            self.send_error(404, "Not Found")
    
    def do_PUT(self):
        if self.path.startswith('/api/'):
            self.proxy_request()
        else:
            self.send_error(404, "Not Found")
    
    def proxy_request(self):
        """Proxy API requests to the backend server"""
        # Remove /api prefix for backend compatibility
        backend_path = self.path.replace('/api', '', 1)
        backend_url = f"http://localhost:8080{backend_path}"
        
        try:
            # Read request body if any
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else None
            
            # Create request
            req = urllib.request.Request(backend_url, data=post_data)
            
            # Copy headers
            for header, value in self.headers.items():
                if header.lower() not in ['host', 'content-length']:
                    req.add_header(header, value)
            
            # Add CORS headers
            req.add_header('Origin', 'http://localhost:3000')
            req.add_header('Access-Control-Request-Method', self.command)
            req.add_header('Access-Control-Request-Headers', 'Content-Type')
            
            # Make request
            with urllib.request.urlopen(req) as response:
                # Send response
                self.send_response(response.getcode())
                
                # Copy response headers
                for header, value in response.headers.items():
                    if header.lower() not in ['server', 'date']:
                        self.send_header(header, value)
                
                # Add CORS headers
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                
                self.end_headers()
                
                # Send response body
                self.wfile.write(response.read())
                
        except Exception as e:
            print(f"Proxy error: {e}")
            self.send_error(500, f"Proxy error: {str(e)}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

def main():
    PORT = 3000
    frontend_dir = Path(__file__).parent
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    print(f"Starting HybridVectorDB Frontend Server")
    print(f"Serving directory: {frontend_dir}")
    print(f"Frontend URL: http://localhost:{PORT}")
    print(f"Backend API: http://localhost:8080")
    print(f"Dashboard: http://localhost:{PORT}")
    print()
    print("Features:")
    print("   • Static HTML dashboard (no changes made)")
    print("   • API proxy to backend server")
    print("   • CORS enabled for cross-origin requests")
    print("   • Live data integration")
    print()
    print("Make sure the backend server is running on port 8080")
    print("   Run: python simple_server.py")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    with socketserver.TCPServer(("", PORT), HybridVectorDBHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    main()
