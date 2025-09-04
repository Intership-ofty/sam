import React, { useState, useEffect } from 'react'
import { api, Alert } from '../../services/api'
import LoadingSpinner from '../../components/Common/LoadingSpinner'

const AlertsPage: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [filteredAlerts, setFilteredAlerts] = useState<Alert[]>([])
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    severity: '',
    status: '',
    search: ''
  })
  const [selectedAlerts, setSelectedAlerts] = useState<Set<string>>(new Set())

  useEffect(() => {
    loadAlerts()
  }, [])

  useEffect(() => {
    filterAlerts()
  }, [alerts, filters])

  const loadAlerts = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await api.getKPIAlerts({ limit: 100 })
      const alertsData = response.data

      setAlerts(alertsData)

    } catch (err) {
      console.error('Failed to load alerts:', err)
      setError('Failed to load alerts. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const filterAlerts = () => {
    let filtered = alerts

    if (filters.severity) {
      filtered = filtered.filter(alert => alert.severity === filters.severity)
    }

    if (filters.status) {
      filtered = filtered.filter(alert => alert.status === filters.status)
    }

    if (filters.search) {
      const searchLower = filters.search.toLowerCase()
      filtered = filtered.filter(alert => 
        alert.kpi_name.toLowerCase().includes(searchLower) ||
        alert.message.toLowerCase().includes(searchLower) ||
        (alert.site_id && alert.site_id.toLowerCase().includes(searchLower))
      )
    }

    setFilteredAlerts(filtered)
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100 border-red-200'
      case 'major': return 'text-orange-600 bg-orange-100 border-orange-200'
      case 'minor': return 'text-yellow-600 bg-yellow-100 border-yellow-200'
      case 'warning': return 'text-blue-600 bg-blue-100 border-blue-200'
      default: return 'text-gray-600 bg-gray-100 border-gray-200'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-red-600 bg-red-100'
      case 'acknowledged': return 'text-yellow-600 bg-yellow-100'
      case 'resolved': return 'text-green-600 bg-green-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return '🔴'
      case 'major': return '🟠'
      case 'minor': return '🟡'
      case 'warning': return '🔵'
      default: return '⚪'
    }
  }

  const handleSelectAlert = (alertId: string) => {
    const newSelected = new Set(selectedAlerts)
    if (newSelected.has(alertId)) {
      newSelected.delete(alertId)
    } else {
      newSelected.add(alertId)
    }
    setSelectedAlerts(newSelected)
  }

  const handleSelectAll = () => {
    if (selectedAlerts.size === filteredAlerts.length) {
      setSelectedAlerts(new Set())
    } else {
      setSelectedAlerts(new Set(filteredAlerts.map(alert => alert.id)))
    }
  }

  const handleBulkAction = (action: string) => {
    // Implement bulk actions (acknowledge, resolve, etc.)
    console.log(`Bulk action: ${action} on alerts:`, Array.from(selectedAlerts))
    setSelectedAlerts(new Set())
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner label="Loading alerts..." />
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
                onClick={loadAlerts}
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
            <h1 className="text-2xl font-bold text-gray-900">Alerts</h1>
            <p className="mt-1 text-sm text-gray-500">
              Monitor and manage system alerts and notifications
            </p>
          </div>
          <button
            onClick={loadAlerts}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Filters and Bulk Actions */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Severity</label>
              <select
                value={filters.severity}
                onChange={(e) => setFilters({ ...filters, severity: e.target.value })}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Severities</option>
                <option value="critical">Critical</option>
                <option value="major">Major</option>
                <option value="minor">Minor</option>
                <option value="warning">Warning</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
              <select
                value={filters.status}
                onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Status</option>
                <option value="active">Active</option>
                <option value="acknowledged">Acknowledged</option>
                <option value="resolved">Resolved</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
              <input
                type="text"
                value={filters.search}
                onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                placeholder="Search alerts..."
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          
          {selectedAlerts.size > 0 && (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500">
                {selectedAlerts.size} selected
              </span>
              <button
                onClick={() => handleBulkAction('acknowledge')}
                className="bg-yellow-100 text-yellow-800 px-3 py-2 rounded-md hover:bg-yellow-200 transition-colors"
              >
                Acknowledge
              </button>
              <button
                onClick={() => handleBulkAction('resolve')}
                className="bg-green-100 text-green-800 px-3 py-2 rounded-md hover:bg-green-200 transition-colors"
              >
                Resolve
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Alerts List */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center">
            <input
              type="checkbox"
              checked={selectedAlerts.size === filteredAlerts.length && filteredAlerts.length > 0}
              onChange={handleSelectAll}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <span className="ml-2 text-sm font-medium text-gray-900">
              Select All ({filteredAlerts.length} alerts)
            </span>
          </div>
        </div>

        <div className="divide-y divide-gray-200">
          {filteredAlerts.map((alert) => (
            <div key={alert.id} className="p-6 hover:bg-gray-50">
              <div className="flex items-start space-x-4">
                <input
                  type="checkbox"
                  checked={selectedAlerts.has(alert.id)}
                  onChange={() => handleSelectAlert(alert.id)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded mt-1"
                />
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{getSeverityIcon(alert.severity)}</span>
                      <div>
                        <h3 className="text-lg font-medium text-gray-900">{alert.kpi_name}</h3>
                        {alert.site_id && (
                          <p className="text-sm text-gray-500">Site: {alert.site_id}</p>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getSeverityColor(alert.severity)}`}>
                        {alert.severity}
                      </span>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(alert.status)}`}>
                        {alert.status}
                      </span>
                    </div>
                  </div>

                  <div className="mt-2">
                    <p className="text-sm text-gray-700">{alert.message}</p>
                  </div>

                  <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <span className="text-xs font-medium text-gray-500">Current Value</span>
                      <p className="text-sm text-gray-900">{alert.current_value.toFixed(2)}</p>
                    </div>
                    <div>
                      <span className="text-xs font-medium text-gray-500">Threshold</span>
                      <p className="text-sm text-gray-900">{alert.threshold_value.toFixed(2)}</p>
                    </div>
                    <div>
                      <span className="text-xs font-medium text-gray-500">Triggered</span>
                      <p className="text-sm text-gray-900">
                        {new Date(alert.triggered_at).toLocaleString()}
                      </p>
                    </div>
                  </div>

                  <div className="mt-4 flex items-center space-x-4">
                    <button className="text-sm text-blue-600 hover:text-blue-800">
                      Acknowledge
                    </button>
                    <button className="text-sm text-green-600 hover:text-green-800">
                      Resolve
                    </button>
                    <button className="text-sm text-gray-600 hover:text-gray-800">
                      View Details
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Empty State */}
      {filteredAlerts.length === 0 && (
        <div className="bg-white shadow rounded-lg p-12 text-center">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">No alerts found</h3>
          <p className="mt-1 text-sm text-gray-500">
            {Object.values(filters).some(f => f) 
              ? 'Try adjusting your filters to see more results.'
              : 'No alerts are currently active.'
            }
          </p>
        </div>
      )}
    </div>
  )
}

export default AlertsPage