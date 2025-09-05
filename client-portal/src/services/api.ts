const API_BASE_URL = 'http://localhost:8000'

// API Response types
interface ApiResponse<T> {
  data: T
  message?: string
  error?: string
}

interface KPIMetric {
  kpi_name: string
  current_value: number
  target_value?: number
  unit: string
  category: string
  trend: 'up' | 'down' | 'stable'
  quality_score: number
  last_calculated: string
  prediction_7d?: number
  metadata: Record<string, any>
}

interface Site {
  site_id: string
  site_code: string
  site_name: string
  latitude: number
  longitude: number
  address: string
  region: string
  country: string
  technology: Record<string, any>
  status: string
  health_score?: number
}

interface SiteHealth {
  site_id: string
  health_score: number
  status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical'
  active_alerts: number
  recent_metrics: number
  categories: Record<string, { kpi_count: number; avg_quality: number }>
  last_updated: string
}

interface Alert {
  id: string
  kpi_name: string
  site_id?: string
  condition_type: string
  threshold_value: number
  current_value: number
  severity: 'critical' | 'major' | 'minor' | 'warning'
  status: 'active' | 'acknowledged' | 'resolved'
  triggered_at: string
  message: string
  metadata: Record<string, any>
}

// API Client class
class ApiClient {
  private baseUrl: string
  private token: string | null = null

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
    this.token = localStorage.getItem('auth_token')
  }

  setToken(token: string) {
    this.token = token
    localStorage.setItem('auth_token', token)
  }

  clearToken() {
    this.token = null
    localStorage.removeItem('auth_token')
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    }

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return { data }
    } catch (error) {
      console.error('API request failed:', error)
      throw error
    }
  }

  // Authentication
  async getAuthConfig() {
    return this.request('/api/v1/health')
  }

  async getCurrentUser() {
    return this.request('/api/v1/auth/me')
  }

  // KPIs
  async getKPIMetrics(params?: {
    site_id?: string
    category?: string
    time_range?: string
    limit?: number
  }) {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }
    
    const queryString = searchParams.toString()
    return this.request<KPIMetric[]>(`/api/v1/kpi/metrics${queryString ? `?${queryString}` : ''}`)
  }

  async getKPITrend(kpiName: string, params?: {
    site_id?: string
    time_range?: string
    resolution?: string
    include_predictions?: boolean
  }) {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }
    
    const queryString = searchParams.toString()
    return this.request(`/api/v1/kpi/trends/${kpiName}${queryString ? `?${queryString}` : ''}`)
  }

  async getKPIAlerts(params?: {
    severity?: string
    status?: string
    site_id?: string
    limit?: number
  }) {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }
    
    const queryString = searchParams.toString()
    return this.request<Alert[]>(`/api/v1/kpi/alerts${queryString ? `?${queryString}` : ''}`)
  }

  // Sites
  async getSites(params?: {
    region?: string
    technology?: string
    status?: string
    limit?: number
    offset?: number
  }) {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }
    
    const queryString = searchParams.toString()
    return this.request<Site[]>(`/api/v1/sites/${queryString ? `?${queryString}` : ''}`)
  }

  async getSite(siteId: string) {
    return this.request<Site>(`/api/v1/sites/${siteId}`)
  }

  async getSiteHealth(siteId: string) {
    return this.request<SiteHealth>(`/api/v1/sites/${siteId}/health`)
  }

  async getSiteKPIs(siteId: string, category?: string) {
    const searchParams = new URLSearchParams()
    if (category) {
      searchParams.append('category', category)
    }
    
    const queryString = searchParams.toString()
    return this.request(`/api/v1/sites/${siteId}/kpis/latest${queryString ? `?${queryString}` : ''}`)
  }

  async getSiteAlerts(siteId: string, params?: {
    severity?: string
    limit?: number
  }) {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }
    
    const queryString = searchParams.toString()
    return this.request<Alert[]>(`/api/v1/sites/${siteId}/alerts/active${queryString ? `?${queryString}` : ''}`)
  }

  // Real-time streaming
  createKPISubscription(kpiName: string, siteId?: string, interval: number = 30) {
    const params = new URLSearchParams()
    if (siteId) {
      params.append('site_id', siteId)
    }
    params.append('interval', interval.toString())
    
    const queryString = params.toString()
    return new EventSource(`${this.baseUrl}/api/v1/kpi/stream/${kpiName}?${queryString}`)
  }
}

// Export singleton instance
export const api = new ApiClient(API_BASE_URL)

// Export types
export type { KPIMetric, Site, SiteHealth, Alert, ApiResponse }