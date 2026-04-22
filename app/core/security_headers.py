from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Cyber Security Enhancement: 
    Injects strict HTTP response headers globally across the application.
    This protects against Cross-Site Scripting (XSS), Clickjacking, 
    and Data Sniffing vulnerabilities.
    
    System Design:
    Implemented as a middleware pattern to decouple security policy 
    enforcement from core business logic routing.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Prevents browsers from guessing the MIME type (protects against drive-by downloads)
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Prevents the site from being rendered inside an iframe (protects against Clickjacking)
        response.headers["X-Frame-Options"] = "DENY"
        # Enables the Cross-Site Scripting (XSS) filter built into most recent web browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Enforces secure (HTTP over SSL/TLS) connections to the server
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Controls resources the user agent is allowed to load for a given page
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: https:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
        
        return response
