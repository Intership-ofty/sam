// API service for HTML frontend
class ApiService {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...window.authService.getAuthHeaders(),
            ...options.headers
        };

        try {
            const response = await fetch(url, {
                ...options,
                headers
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // KPIs
    async getKPIMetrics(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/api/v1/kpi/metrics${queryString ? `?${queryString}` : ''}`);
    }

    async getKPITrend(kpiName, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/api/v1/kpi/trends/${kpiName}${queryString ? `?${queryString}` : ''}`);
    }

    async getKPIAlerts(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/api/v1/kpi/alerts${queryString ? `?${queryString}` : ''}`);
    }

    // Sites
    async getSites(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/api/v1/sites/${queryString ? `?${queryString}` : ''}`);
    }

    async getSite(siteId) {
        return this.request(`/api/v1/sites/${siteId}`);
    }

    async getSiteHealth(siteId) {
        return this.request(`/api/v1/sites/${siteId}/health`);
    }

    async getSiteKPIs(siteId, category = null) {
        const params = category ? { category } : {};
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/api/v1/sites/${siteId}/kpis/latest${queryString ? `?${queryString}` : ''}`);
    }

    async getSiteAlerts(siteId, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/api/v1/sites/${siteId}/alerts/active${queryString ? `?${queryString}` : ''}`);
    }

    // Health check
    async getHealth() {
        return this.request('/health');
    }

    // Create KPI subscription for real-time data
    createKPISubscription(kpiName, siteId = null, interval = 30) {
        const params = new URLSearchParams();
        if (siteId) params.append('site_id', siteId);
        params.append('interval', interval.toString());
        
        return new EventSource(`${this.baseUrl}/api/v1/kpi/stream/${kpiName}?${params.toString()}`);
    }
}

// Global API instance
window.apiService = new ApiService();
