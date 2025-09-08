import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { TenantProvider } from './contexts/TenantContext'
import { ThemeProvider } from './contexts/ThemeContext'
import Sidebar from './components/Layout/Sidebar'
import Header from './components/Layout/Header'
import LoadingSpinner from './components/Common/LoadingSpinner'
import './styles.css'

import LoginPage from './pages/Auth/LoginPage'
import DashboardPage from './pages/Dashboard/DashboardPage'
import SitesPage from './pages/Sites/SitesPage'
import SiteDetailPage from './pages/Sites/SiteDetailPage'
import KPIsPage from './pages/KPIs/KPIsPage'
import KPIManagementPage from './pages/KPIs/KPIManagementPage'
import AlertsPage from './pages/Alerts/AlertsPage'
import ReportsPage from './pages/Reports/ReportsPage'
import SLAPage from './pages/SLA/SLAPage'
import MaintenancePage from './pages/Maintenance/MaintenancePage'
import SettingsPage from './pages/Settings/SettingsPage'

function Protected({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, loading } = useAuth()
  if (loading) return <div className="center"><LoadingSpinner label="Loading session..." /></div>
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return <>{children}</>
}

export default function Root() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <TenantProvider>
          <div className="layout">
            <Sidebar />
            <div className="main">
              <Header />
              <div className="content">
                <Routes>
                  <Route path="/login" element={<LoginPage />} />
                  <Route path="/" element={<Protected><DashboardPage /></Protected>} />
                  <Route path="/sites" element={<Protected><SitesPage /></Protected>} />
                  <Route path="/sites/:id" element={<Protected><SiteDetailPage /></Protected>} />
                  <Route path="/kpis" element={<Protected><KPIsPage /></Protected>} />
                  <Route path="/kpis/management" element={<Protected><KPIManagementPage /></Protected>} />
                  <Route path="/alerts" element={<Protected><AlertsPage /></Protected>} />
                  <Route path="/reports" element={<Protected><ReportsPage /></Protected>} />
                  <Route path="/sla" element={<Protected><SLAPage /></Protected>} />
                  <Route path="/maintenance" element={<Protected><MaintenancePage /></Protected>} />
                  <Route path="/settings" element={<Protected><SettingsPage /></Protected>} />
                  <Route path="*" element={<Navigate to="/" />} />
                </Routes>
              </div>
            </div>
          </div>
        </TenantProvider>
      </AuthProvider>
    </ThemeProvider>
  )
}
