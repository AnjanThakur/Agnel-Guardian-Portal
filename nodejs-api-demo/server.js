const http = require('http');
const jwt = require('jsonwebtoken');
const fs = require('fs');
const path = require('path');

const JWT_SECRET = 'super-secret-native-node-key';

// Helper function to read request body
const getRequestBody = (req) => {
    return new Promise((resolve, reject) => {
        let body = '';
        req.on('data', chunk => body += chunk.toString());
        req.on('end', () => {
            try {
                resolve(body ? JSON.parse(body) : {});
            } catch (err) {
                reject(err);
            }
        });
    });
};

// 1. Authorization Middleware (Verifying JWT)
const verifyToken = (req, res) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Format: Bearer <token>

    if (!token) {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Access Denied: No token provided' }));
        return null;
    }

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        return decoded; // User is authenticated
    } catch (err) {
        res.writeHead(403, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Access Denied: Invalid or expired token' }));
        return null;
    }
};

// Main HTTP Server (RESTful Routing)
const server = http.createServer(async (req, res) => {
    
    // Serve the UI (index.html)
    if (req.method === 'GET' && req.url === '/') {
        const filePath = path.join(__dirname, 'index.html');
        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(500);
                res.end("Error loading UI");
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content, 'utf-8');
            }
        });
    }
    // 2. Authentication: Login endpoint (Generating JWT)
    else if (req.method === 'POST' && req.url === '/api/auth/login') {
        try {
            const body = await getRequestBody(req);
            
            if (body.username && body.password) {
                // Generate secure Token
                const token = jwt.sign(
                    { username: body.username, role: 'node-user' }, 
                    JWT_SECRET, 
                    { expiresIn: '1h' }
                );
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ message: "Login successful", token: token }));
            } else {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: "Please provide username and password" }));
            }
        } catch (error) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: "Invalid JSON format" }));
        }

    } 
    // 3. Protected RESTful API Route
    else if (req.method === 'GET' && req.url === '/api/protected-data') {
        const user = verifyToken(req, res);
        
        // If verifyToken returned user, auth was successful
        if (user) {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                message: "Success! You have accessed the protected API via Native Node.js.",
                user: user,
                data: [100, 200, 300]
            }));
        }
    } 
    // Handle 404
    else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: "Route not found" }));
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Native Node.js REST API running on http://localhost:${PORT}`);
});
