"""
Production-ready Flask application (Backward Compatibility Wrapper)
This file maintains backward compatibility while using the new app structure.
"""

from app import app

if __name__ == '__main__':
    # Models are already loaded at module level
    # Run with development server (use Gunicorn in production)
    app.run(host="0.0.0.0", port=5000, debug=False)
