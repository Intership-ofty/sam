import React, { useState, useEffect } from 'react'
import { api, type KPIMetric } from '../../services/api'

interface KPIDefinition {
  name: string
  display_name: string
  description: string
  category: 'network' | 'energy' | 'operational' | 'financial'
  unit: string
  formula: string
  target_value?: number
  warning_threshold?: number
  critical_threshold?: number
  calculation_interval: string
  enabled: boolean
  tenant_specific: boolean
  metadata: Record<string, any>
}

const KPIManagementPage: React.FC = () => {
  const [kpis, setKpis] = useState<KPIDefinition[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [editingKpi, setEditingKpi] = useState<KPIDefinition | null>(null)
  const [formData, setFormData] = useState<KPIDefinition>({
    name: '',
    display_name: '',
    description: '',
    category: 'network',
    unit: '',
    formula: '',
    target_value: undefined,
    warning_threshold: undefined,
    critical_threshold: undefined,
    calculation_interval: '5m',
    enabled: true,
    tenant_specific: false,
    metadata: {}
  })

  useEffect(() => {
    loadKPIs()
  }, [])

  const loadKPIs = async () => {
    try {
      setLoading(true)
      const response = await api.getKPIDefinitions()
      setKpis(response.data || [])
    } catch (error) {
      console.error('Error loading KPIs:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      if (editingKpi) {
        await api.updateKPIDefinition(editingKpi.name, formData)
        await loadKPIs()
        setShowAddForm(false)
        resetForm()
      } else {
        await api.createKPIDefinition(formData)
        await loadKPIs()
        setShowAddForm(false)
        resetForm()
      }
    } catch (error) {
      console.error('Error saving KPI:', error)
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      display_name: '',
      description: '',
      category: 'network',
      unit: '',
      formula: '',
      target_value: undefined,
      warning_threshold: undefined,
      critical_threshold: undefined,
      calculation_interval: '5m',
      enabled: true,
      tenant_specific: false,
      metadata: {}
    })
    setEditingKpi(null)
  }

  const handleEdit = (kpi: KPIDefinition) => {
    setEditingKpi(kpi)
    setFormData(kpi)
    setShowAddForm(true)
  }

  const handleDelete = async (kpiName: string) => {
    if (window.confirm('Êtes-vous sûr de vouloir supprimer ce KPI ?')) {
      try {
        await api.deleteKPIDefinition(kpiName)
        await loadKPIs()
      } catch (error) {
        console.error('Error deleting KPI:', error)
      }
    }
  }

  const categories = [
    { value: 'network', label: 'Réseau', color: 'bg-blue-100 text-blue-800' },
    { value: 'energy', label: 'Énergie', color: 'bg-green-100 text-green-800' },
    { value: 'operational', label: 'Opérationnel', color: 'bg-purple-100 text-purple-800' },
    { value: 'financial', label: 'Financier', color: 'bg-yellow-100 text-yellow-800' }
  ]

  const intervals = [
    { value: '1m', label: '1 minute' },
    { value: '5m', label: '5 minutes' },
    { value: '15m', label: '15 minutes' },
    { value: '30m', label: '30 minutes' },
    { value: '1h', label: '1 heure' },
    { value: '6h', label: '6 heures' },
    { value: '1d', label: '1 jour' }
  ]

  if (loading) {
    return (
      <div className="center">
        <div className="loading-spinner"></div>
        <p className="mt-4 text-gray-600">Chargement des KPIs...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Gestion des KPIs</h1>
            <p className="mt-1 text-sm text-gray-500">
              Créez et gérez vos indicateurs de performance personnalisés
            </p>
          </div>
          <button
            onClick={() => setShowAddForm(true)}
            className="btn-primary"
          >
            <span className="mr-2">+</span>
            Nouveau KPI
          </button>
        </div>
      </div>

      {/* KPI List */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">KPIs existants ({kpis.length})</h2>
        </div>
        
        {kpis.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-gray-400 text-6xl mb-4">📊</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Aucun KPI défini</h3>
            <p className="text-gray-500 mb-4">Commencez par créer votre premier KPI personnalisé</p>
            <button
              onClick={() => setShowAddForm(true)}
              className="btn-primary"
            >
              Créer un KPI
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th>Nom</th>
                  <th>Catégorie</th>
                  <th>Unité</th>
                  <th>Formule</th>
                  <th>Intervalle</th>
                  <th>Statut</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {kpis.map((kpi) => {
                  const category = categories.find(c => c.value === kpi.category)
                  return (
                    <tr key={kpi.name}>
                      <td>
                        <div>
                          <div className="font-medium text-gray-900">{kpi.display_name}</div>
                          <div className="text-sm text-gray-500">{kpi.name}</div>
                        </div>
                      </td>
                      <td>
                        <span className={`status-indicator ${category?.color}`}>
                          {category?.label}
                        </span>
                      </td>
                      <td className="text-sm text-gray-900">{kpi.unit}</td>
                      <td className="text-sm text-gray-900 font-mono">{kpi.formula}</td>
                      <td className="text-sm text-gray-900">{kpi.calculation_interval}</td>
                      <td>
                        <span className={`status-indicator ${kpi.enabled ? 'status-active' : 'status-inactive'}`}>
                          {kpi.enabled ? 'Actif' : 'Inactif'}
                        </span>
                      </td>
                      <td>
                        <div className="flex space-x-2">
                          <button
                            onClick={() => handleEdit(kpi)}
                            className="text-blue-600 hover:text-blue-800 text-sm"
                          >
                            Modifier
                          </button>
                          <button
                            onClick={() => handleDelete(kpi.name)}
                            className="text-red-600 hover:text-red-800 text-sm"
                          >
                            Supprimer
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Add/Edit Form Modal */}
      {showAddForm && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-900">
                  {editingKpi ? 'Modifier le KPI' : 'Nouveau KPI'}
                </h3>
                <button
                  onClick={() => {
                    setShowAddForm(false)
                    resetForm()
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">Nom unique *</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      className="w-full"
                      placeholder="ex: custom_network_quality"
                      required
                    />
                  </div>
                  <div>
                    <label className="label">Nom affiché *</label>
                    <input
                      type="text"
                      value={formData.display_name}
                      onChange={(e) => setFormData({...formData, display_name: e.target.value})}
                      className="w-full"
                      placeholder="ex: Qualité réseau personnalisée"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="label">Description *</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({...formData, description: e.target.value})}
                    className="w-full"
                    rows={3}
                    placeholder="Description détaillée du KPI"
                    required
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">Catégorie *</label>
                    <select
                      value={formData.category}
                      onChange={(e) => setFormData({...formData, category: e.target.value as any})}
                      className="w-full"
                      required
                    >
                      {categories.map(cat => (
                        <option key={cat.value} value={cat.value}>{cat.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Unité de mesure *</label>
                    <input
                      type="text"
                      value={formData.unit}
                      onChange={(e) => setFormData({...formData, unit: e.target.value})}
                      className="w-full"
                      placeholder="ex: %, dBm, Mbps"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="label">Formule de calcul *</label>
                  <input
                    type="text"
                    value={formData.formula}
                    onChange={(e) => setFormData({...formData, formula: e.target.value})}
                    className="w-full font-mono"
                    placeholder="ex: AVG(signal_quality_pct)"
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Utilisez des expressions SQL (AVG, SUM, COUNT, etc.)
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="label">Valeur cible</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.target_value || ''}
                      onChange={(e) => setFormData({...formData, target_value: e.target.value ? parseFloat(e.target.value) : undefined})}
                      className="w-full"
                      placeholder="95.0"
                    />
                  </div>
                  <div>
                    <label className="label">Seuil d'avertissement</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.warning_threshold || ''}
                      onChange={(e) => setFormData({...formData, warning_threshold: e.target.value ? parseFloat(e.target.value) : undefined})}
                      className="w-full"
                      placeholder="90.0"
                    />
                  </div>
                  <div>
                    <label className="label">Seuil critique</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.critical_threshold || ''}
                      onChange={(e) => setFormData({...formData, critical_threshold: e.target.value ? parseFloat(e.target.value) : undefined})}
                      className="w-full"
                      placeholder="85.0"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">Intervalle de calcul</label>
                    <select
                      value={formData.calculation_interval}
                      onChange={(e) => setFormData({...formData, calculation_interval: e.target.value})}
                      className="w-full"
                    >
                      {intervals.map(interval => (
                        <option key={interval.value} value={interval.value}>{interval.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex items-center space-x-4">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={formData.enabled}
                        onChange={(e) => setFormData({...formData, enabled: e.target.checked})}
                        className="mr-2"
                      />
                      Actif
                    </label>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={formData.tenant_specific}
                        onChange={(e) => setFormData({...formData, tenant_specific: e.target.checked})}
                        className="mr-2"
                      />
                      Spécifique au tenant
                    </label>
                  </div>
                </div>

                <div className="flex justify-end space-x-3 pt-4">
                  <button
                    type="button"
                    onClick={() => {
                      setShowAddForm(false)
                      resetForm()
                    }}
                    className="btn-secondary"
                  >
                    Annuler
                  </button>
                  <button
                    type="submit"
                    className="btn-primary"
                  >
                    {editingKpi ? 'Modifier' : 'Créer'} le KPI
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default KPIManagementPage
