// Authentication service for HTML frontend
class AuthService {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.token = localStorage.getItem('auth_token');
        this.user = null;
    }

    async login() {
        try {
            // Get auth config
            const configResponse = await fetch(`${this.baseUrl}/api/v1/auth/config`);
            const config = await configResponse.json();
            
            if (config.data.auth_mode === 'keycloak') {
                // Redirect to Keycloak login
                const loginResponse = await fetch(`${this.baseUrl}/api/v1/auth/login`);
                const loginData = await loginResponse.json();
                
                // Redirect to Keycloak
                window.location.href = loginData.login_url;
            } else {
                // Simple JWT auth for development
                this.token = 'dev-token-' + Date.now();
                localStorage.setItem('auth_token', this.token);
                this.user = { name: 'Developer', email: 'dev@towerco.com' };
                return true;
            }
        } catch (error) {
            console.error('Login failed:', error);
            return false;
        }
    }

    async getCurrentUser() {
        if (!this.token) return null;
        
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.user = data.user;
                return this.user;
            }
        } catch (error) {
            console.error('Failed to get user:', error);
        }
        
        return null;
    }

    logout() {
        this.token = null;
        this.user = null;
        localStorage.removeItem('auth_token');
        window.location.href = 'index.html';
    }

    isAuthenticated() {
        return !!this.token;
    }

    getAuthHeaders() {
        return this.token ? {
            'Authorization': `Bearer ${this.token}`,
            'Content-Type': 'application/json'
        } : {
            'Content-Type': 'application/json'
        };
    }
}

// Global auth instance
window.authService = new AuthService();
