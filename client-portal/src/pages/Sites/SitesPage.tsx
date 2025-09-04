import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, Site, SiteHealth } from '../../services/api'
import LoadingSpinner from '../../components/Common/LoadingSpinner'

const SitesPage: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [sites, setSites] = useState<Site[]>([])
  const [siteHealth, setSiteHealth] = useState<Record<string, SiteHealth>>({})
  const [filteredSites, setFilteredSites] = useState<Site[]>([])
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    region: '',
    technology: '',
    status: '',
    search: ''
  })
  const [regions, setRegions] = useState<string[]>([])
  const [technologies, setTechnologies] = useState<string[]>([])

  useEffect(() => {
    loadSites()
  }, [])

  useEffect(() => {
    filterSites()
  }, [sites, filters])

  const loadSites = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await api.getSites({ limit: 100 })
      const sitesData = response.data

      setSites(sitesData)
      
      // Extract unique regions and technologies
      const uniqueRegions = [...new Set(sitesData.map(site => site.region).filter(Boolean))]
      const uniqueTechnologies = [...new Set(
        sitesData.flatMap(site => Object.keys(site.technology || {}))
      )]
      
      setRegions(uniqueRegions)
      setTechnologies(uniqueTechnologies)

      // Load health data for each site
      const healthPromises = sitesData.map(async (site) => {
        try {
          const healthResponse = await api.getSiteHealth(site.site_id)
          return { siteId: site.site_id, health: healthResponse.data }
        } catch (err) {
          console.warn(`Failed to load health for site ${site.site_id}:`, err)
          return null
        }
      })

      const healthResults = await Promise.all(healthPromises)
      const healthMap: Record<string, SiteHealth> = {}
      
      healthResults.forEach(result => {
        if (result) {
          healthMap[result.siteId] = result.health
        }
      })

      setSiteHealth(healthMap)

    } catch (err) {
      console.error('Failed to load sites:', err)
      setError('Failed to load sites. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const filterSites = () => {
    let filtered = sites

    if (filters.region) {
      filtered = filtered.filter(site => site.region === filters.region)
    }

    if (filters.technology) {
      filtered = filtered.filter(site => 
        site.technology && Object.keys(site.technology).includes(filters.technology)
      )
    }

    if (filters.status) {
      filtered = filtered.filter(site => site.status === filters.status)
    }

    if (filters.search) {
      const searchLower = filters.search.toLowerCase()
      filtered = filtered.filter(site => 
        site.site_name.toLowerCase().includes(searchLower) ||
        site.site_code.toLowerCase().includes(searchLower) ||
        site.address.toLowerCase().includes(searchLower)
      )
    }

    setFilteredSites(filtered)
  }

  const getHealthStatusColor = (score: number) => {
    if (score >= 90) return 'text-green-600 bg-green-100'
    if (score >= 70) return 'text-yellow-600 bg-yellow-100'
    if (score >= 50) return 'text-orange-600 bg-orange-100'
    return 'text-red-600 bg-red-100'
  }

  const getHealthStatusText = (score: number) => {
    if (score >= 90) return 'Excellent'
    if (score >= 70) return 'Good'
    if (score >= 50) return 'Fair'
    if (score >= 30) return 'Poor'
    return 'Critical'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800'
      case 'inactive': return 'bg-gray-100 text-gray-800'
      case 'maintenance': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner label="Loading sites..." />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
            <div className="mt-4">
              <button
                onClick={loadSites}
                className="bg-red-100 px-3 py-2 rounded-md text-sm font-medium text-red-800 hover:bg-red-200"
              >
                Try Again
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Sites</h1>
            <p className="mt-1 text-sm text-gray-500">
              Manage and monitor your telecom infrastructure sites
            </p>
          </div>
          <button
            onClick={loadSites}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Search</label>
            <input
              type="text"
              value={filters.search}
              onChange={(e) => setFilters({ ...filters, search: e.target.value })}
              placeholder="Search sites..."
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Region</label>
            <select
              value={filters.region}
              onChange={(e) => setFilters({ ...filters, region: e.target.value })}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Regions</option>
              {regions.map(region => (
                <option key={region} value={region}>{region}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Technology</label>
            <select
              value={filters.technology}
              onChange={(e) => setFilters({ ...filters, technology: e.target.value })}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Technologies</option>
              {technologies.map(tech => (
                <option key={tech} value={tech}>{tech}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
            <select
              value={filters.status}
              onChange={(e) => setFilters({ ...filters, status: e.target.value })}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Status</option>
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
              <option value="maintenance">Maintenance</option>
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={() => setFilters({ region: '', technology: '', status: '', search: '' })}
              className="w-full bg-gray-100 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-200 transition-colors"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>

      {/* Sites Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredSites.map((site) => {
          const health = siteHealth[site.site_id]
          return (
            <div key={site.site_id} className="bg-white shadow rounded-lg p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900 truncate">{site.site_name}</h3>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(site.status)}`}>
                  {site.status}
                </span>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Site Code</span>
                  <span className="text-sm font-medium text-gray-900">{site.site_code}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Region</span>
                  <span className="text-sm font-medium text-gray-900">{site.region}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Country</span>
                  <span className="text-sm font-medium text-gray-900">{site.country}</span>
                </div>

                {health && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-500">Health Score</span>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getHealthStatusColor(health.health_score * 100)}`}>
                        {(health.health_score * 100).toFixed(0)}%
                      </span>
                      <span className="text-xs text-gray-500">
                        {getHealthStatusText(health.health_score * 100)}
                      </span>
                    </div>
                  </div>
                )}

                {health && health.active_alerts > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-500">Active Alerts</span>
                    <span className="text-sm font-medium text-red-600">{health.active_alerts}</span>
                  </div>
                )}

                <div className="pt-2 border-t border-gray-200">
                  <p className="text-xs text-gray-500 truncate">{site.address}</p>
                </div>

                <div className="pt-2">
                  <Link
                    to={`/sites/${site.site_id}`}
                    className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors text-center block"
                  >
                    View Details
                  </Link>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Empty State */}
      {filteredSites.length === 0 && (
        <div className="bg-white shadow rounded-lg p-12 text-center">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">No sites found</h3>
          <p className="mt-1 text-sm text-gray-500">
            {Object.values(filters).some(f => f) 
              ? 'Try adjusting your filters to see more results.'
              : 'No sites are available at the moment.'
            }
          </p>
        </div>
      )}
    </div>
  )
}

export default SitesPage