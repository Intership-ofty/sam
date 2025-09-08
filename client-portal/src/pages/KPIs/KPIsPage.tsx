import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, KPIMetric } from '../../services/api'
import LoadingSpinner from '../../components/Common/LoadingSpinner'

const KPIsPage: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [kpis, setKpis] = useState<KPIMetric[]>([])
  const [filteredKpis, setFilteredKpis] = useState<KPIMetric[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [selectedSite, setSelectedSite] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const [categories, setCategories] = useState<string[]>([])

  useEffect(() => {
    loadKPIs()
  }, [])

  useEffect(() => {
    filterKpis()
  }, [kpis, selectedCategory, selectedSite])

  const loadKPIs = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await api.getKPIMetrics({ limit: 100 })
      const kpisData = response.data

      setKpis(kpisData)
      
      // Extract unique categories
      const uniqueCategories = [...new Set(kpisData.map(kpi => kpi.category))]
      setCategories(uniqueCategories)

    } catch (err) {
      console.error('Failed to load KPIs:', err)
      setError('Failed to load KPIs. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const filterKpis = () => {
    let filtered = kpis

    if (selectedCategory) {
      filtered = filtered.filter(kpi => kpi.category === selectedCategory)
    }

    if (selectedSite) {
      filtered = filtered.filter(kpi => kpi.metadata?.site_id === selectedSite)
    }

    setFilteredKpis(filtered)
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return '↗️'
      case 'down': return '↘️'
      default: return '→'
    }
  }

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up': return 'text-green-600'
      case 'down': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getQualityColor = (score: number) => {
    if (score >= 0.9) return 'text-green-600 bg-green-100'
    if (score >= 0.7) return 'text-yellow-600 bg-yellow-100'
    if (score >= 0.5) return 'text-orange-600 bg-orange-100'
    return 'text-red-600 bg-red-100'
  }

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      network: 'bg-blue-100 text-blue-800',
      energy: 'bg-green-100 text-green-800',
      operational: 'bg-yellow-100 text-yellow-800',
      financial: 'bg-purple-100 text-purple-800'
    }
    return colors[category] || 'bg-gray-100 text-gray-800'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner label="Loading KPIs..." />
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
                onClick={loadKPIs}
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
            <h1 className="text-2xl font-bold text-gray-900">KPIs</h1>
            <p className="mt-1 text-sm text-gray-500">
              Key Performance Indicators for your telecom infrastructure
            </p>
          </div>
          <div className="flex space-x-3">
            <Link
              to="/kpis/management"
              className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors"
            >
              <span className="mr-2">⚙️</span>
              Gérer les KPIs
            </Link>
            <button
              onClick={loadKPIs}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Category</label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Categories</option>
              {categories.map(category => (
                <option key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Site</label>
            <select
              value={selectedSite}
              onChange={(e) => setSelectedSite(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Sites</option>
              {/* Add site options here */}
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={() => {
                setSelectedCategory('')
                setSelectedSite('')
              }}
              className="w-full bg-gray-100 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-200 transition-colors"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredKpis.map((kpi) => (
          <div key={kpi.kpi_name} className="bg-white shadow rounded-lg p-6 hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 truncate">{kpi.kpi_name}</h3>
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${getCategoryColor(kpi.category)}`}>
                {kpi.category}
              </span>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Current Value</span>
                <span className="text-2xl font-bold text-gray-900">
                  {kpi.current_value.toFixed(2)} {kpi.unit}
                </span>
              </div>

              {kpi.target_value && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Target</span>
                  <span className="text-sm font-medium text-gray-900">
                    {kpi.target_value.toFixed(2)} {kpi.unit}
                  </span>
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Trend</span>
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{getTrendIcon(kpi.trend)}</span>
                  <span className={`text-sm font-medium ${getTrendColor(kpi.trend)}`}>
                    {kpi.trend}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Quality Score</span>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${getQualityColor(kpi.quality_score)}`}>
                  {(kpi.quality_score * 100).toFixed(0)}%
                </span>
              </div>

              {kpi.prediction_7d && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">7-Day Prediction</span>
                  <span className="text-sm font-medium text-gray-900">
                    {kpi.prediction_7d.toFixed(2)} {kpi.unit}
                  </span>
                </div>
              )}

              <div className="pt-2 border-t border-gray-200">
                <p className="text-xs text-gray-500">
                  Last updated: {new Date(kpi.last_calculated).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {filteredKpis.length === 0 && (
        <div className="bg-white shadow rounded-lg p-12 text-center">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">No KPIs found</h3>
          <p className="mt-1 text-sm text-gray-500">
            {selectedCategory || selectedSite 
              ? 'Try adjusting your filters to see more results.'
              : 'No KPIs are available at the moment.'
            }
          </p>
        </div>
      )}
    </div>
  )
}

export default KPIsPage